# BNN_FCC RTL Design Description

This document summarizes the current RTL implementation in `rtl/` so a follow-on agent can redesign it for higher performance without first rediscovering the architecture. It focuses on:

- What each module does
- How data and control move between modules
- What the current implementation assumes
- Where the existing performance bottlenecks come from

## 1. Project Intent

The RTL implements a fully connected binary neural network (BNN) classifier with three AXI-style streaming interfaces:

- `config_*`: loads weights and thresholds
- `data_in_*`: streams image pixels
- `data_out_*`: emits one classification result per image

The top-level module is `bnn_fcc.sv`. The network topology is parameterized by `TOPOLOGY`, where:

- `TOPOLOGY[0]` is the number of input features
- `TOPOLOGY[1] ... TOPOLOGY[TOTAL_LAYERS-2]` are hidden-layer neuron counts
- `TOPOLOGY[TOTAL_LAYERS-1]` is the output-layer neuron count

Although the README describes a parameterized architecture, the current RTL is much narrower in practice:

- Only one neuron is computed at a time per layer
- Only one image is buffered/in flight at a time
- Layers execute strictly sequentially
- `PARALLEL_INPUTS` is effectively forced to `8`
- `PARALLEL_NEURONS` is not actually implemented

## 2. High-Level Execution Flow

For one complete inference, the RTL behaves like this:

1. Configuration messages arrive on `config_*`.
2. `config_parser.sv` parses the 128-bit header, then forwards payload beats into the selected layer's memory.
3. Each non-input layer has its own `layer_memory.sv` instance that stores weights and thresholds.
4. Image pixels arrive on `data_in_*`.
5. `bnn_fcc.sv` binarizes the pixels and stores the full image into `input_buffer`.
6. Once the full image is captured, layer 1 starts.
7. Each layer's `compute_layer.sv` walks through all neurons sequentially, requesting weight chunks and thresholds from its memory.
8. Each chunk is processed by a single `neural_processor.sv` instance, which performs XNOR-popcount accumulation.
9. Hidden layers threshold their accumulated popcount into a single activation bit per neuron.
10. The output layer stores raw 32-bit scores, not bits.
11. After the final layer completes, `bnn_fcc` performs an argmax over the output scores and emits the class index on `data_out_*`.

This is a batch-serial architecture with no overlap between:

- image buffering and inference
- adjacent layers
- different images
- different neurons within a layer

## 3. Module Roles

### 3.1 `bnn_types_pkg.sv`

`bnn_types_pkg.sv` defines the configuration header format shared by the config path:

- `msg_type_e`
  - `MSG_TYPE_WEIGHTS = 0`
  - `MSG_TYPE_THRESHOLDS = 1`
- `config_header_t`
  - packed 128-bit header matching the README bit layout

This package exists only to make header decoding explicit and type-safe in the config parser.

### 3.2 `config_parser.sv`

`config_parser.sv` is the write-side control block for model loading.

Its job is:

- consume the configuration stream
- deserialize the 128-bit header
- determine whether the payload is weights or thresholds
- determine which layer should receive the payload
- broadcast `wr_addr`, `wr_data`, and `wr_strb` to all layer memories
- assert exactly one per-layer write enable during payload reception

Important behavior:

- Logical layer IDs in the stream are offset by 1 relative to `TOPOLOGY`.
  - Config `layer_id = 0` maps to compute layer 1.
  - This is because `TOPOLOGY[0]` is the input layer and has no memory.
- `layer_wr_addr` increments once per accepted payload beat.
- Payload termination is recognized by either:
  - `config_last`, or
  - the accumulated valid byte count reaching `header.total_bytes`

Current limitations and assumptions:

- Header capture is only implemented for `CONFIG_BUS_WIDTH == 64` or `32`.
- `config_ready` is effectively always high outside reset.
- `layer_wr_strb` is generated, but downstream memory storage does not actually use byte strobes to merge partial words.
- The parser writes raw bus words into memory; it does not repack payloads.

### 3.3 `layer_memory.sv`

`layer_memory.sv` stores one layer's weights and thresholds and serves them to the compute engine during inference.

It has two distinct roles:

- configuration-time storage
- inference-time sequential readback

#### Weight storage strategy

Weights are stored as the incoming packed payload stream, where each neuron's weights are padded to a byte boundary. The memory computes:

- `BYTES_PER_NEURON = ceil(LAYER_INPUTS / 8)`
- `BITS_PER_NEURON = BYTES_PER_NEURON * 8`
- `WEIGHT_MEM_DEPTH = ceil(NUM_NEURONS * BITS_PER_NEURON / CONFIG_BUS_WIDTH)`

Because a requested chunk can start at an arbitrary byte offset inside the packed stream, the module may need two adjacent bus words to reconstruct one compute chunk. To support this with synchronous FPGA-style RAM, the design duplicates the weight memory:

- `mem_weights_lo`
- `mem_weights_hi`

Both arrays contain identical data. On each cycle the module reads:

- the current word from `mem_weights_lo`
- the next word from `mem_weights_hi`

It then concatenates and right-shifts them to produce the aligned `rd_data_weights`.

This duplication is a functional workaround for single-read-port BRAM inference, but it costs roughly 2x weight memory storage.

#### Threshold storage strategy

Thresholds are stored in `mem_thresholds` as raw `CONFIG_BUS_WIDTH` words. On readback, the module slices out one 32-bit threshold using:

- `ptr_thresholds`: word index
- `sub_ptr_thresholds`: 32-bit lane index inside the bus word

#### Read-side sequencing

The memory does not choose which neuron or chunk to read on its own. It advances through data in response to control strobes from `compute_layer`:

- `layer_start`
  - resets inference read pointers to the first neuron and first chunk
- `read_weight_chunk`
  - advances to the next chunk for the current neuron
- `read_threshold`
  - advances to the next neuron and the next threshold

The outputs are prefetched synchronously. In other words:

- the control strobes update the internal pointers at the clock edge
- the corresponding data becomes available on the next cycle

That timing contract is the reason `compute_layer` includes explicit prefetch bubbles.

### 3.4 `neural_processor.sv`

`neural_processor.sv` is the arithmetic core for one neuron stream.

For each `N`-bit chunk:

1. Compute `xnor_result = ~(x ^ w)`
2. Compute `popcount = countones(xnor_result)`
3. Accumulate the popcount across all chunks of the neuron
4. On the final chunk (`last=1`), emit:
   - `valid_out`
   - `y = (accumulated_popcount >= threshold)`
   - `popcount_out = accumulated_popcount`

Within the current design:

- hidden layers ignore `y` and recompute the threshold comparison in `compute_layer`
- the output layer ignores `y` and uses `popcount_out` as the score

So functionally the module is an accumulator with a built-in final compare.

### 3.5 `compute_layer.sv`

`compute_layer.sv` is the per-layer inference controller.

Each instance owns one complete layer execution. It is responsible for:

- requesting data from `layer_memory`
- chunking the input activation vector
- masking off padded bits
- feeding chunks into `neural_processor`
- collecting one result per neuron
- reporting when the full layer is complete

#### Internal operation

For each neuron, the module runs this state sequence:

- `IDLE`
  - wait for `start`
- `PREFETCH`
  - one-cycle bubble so synchronous memory outputs become valid
- `COMPUTE_NEURON`
  - stream one `CONFIG_BUS_WIDTH` chunk per cycle into `neural_processor`
- `FINISH_NEURON`
  - wait for `neural_processor.valid_out`, then store the result
- `DONE_STATE`
  - pulse `done`, then return to `IDLE`

Key details:

- `input_activations` is a bit vector from the previous layer.
- For hidden layers, the output written into `results` is one bit per neuron.
- For the output layer, the output written into `results` is one 32-bit score per neuron.
- Unused padding bits are neutralized before the XNOR:
  - padded activation bits are forced to `0`
  - padded weight bits are forced to `1`
  - this preserves the BNN "no effect" behavior described in the README

#### Read handshake with `layer_memory`

`compute_layer` drives three memory-side control strobes:

- `mem_layer_start`
  - reset the read pointers when a layer starts
- `mem_read_weight`
  - asserted while computing chunks, causing the memory to prefetch the next chunk
- `mem_read_thresh`
  - asserted after a neuron finishes, causing the memory to advance to the next threshold and next neuron's base address

This interface makes the compute engine the owner of the layer traversal order.

#### Result format

`result_vector` is always `NUM_NEURONS * 32` bits wide, but hidden layers only use the low `NUM_NEURONS` bits:

- hidden layers write `results[neuron_cnt]`
- output layer writes `results[neuron_cnt*32 +: 32]`

At the top level, the next layer reads only the low `TOPOLOGY[l-1]` bits from the previous layer's activation storage, so the unused upper bits are ignored.

### 3.6 `bnn_fcc.sv`

`bnn_fcc.sv` is the top-level integrator.

It wires together:

- one global `config_parser`
- one `layer_memory` per compute layer
- one `compute_layer` per compute layer
- image binarization and buffering logic
- final argmax and output handshake logic

Its responsibilities break down as follows.

#### A. Configuration path

`config_parser` broadcasts:

- `layer_wr_addr`
- `layer_wr_data`
- `layer_wr_strb`

to all layer memories, while per-layer enables choose the destination memory.

There is one memory instance per non-input layer:

- layer 1 memory stores weights/thresholds for topology edge `TOPOLOGY[0] -> TOPOLOGY[1]`
- layer 2 memory stores weights/thresholds for `TOPOLOGY[1] -> TOPOLOGY[2]`
- etc.

#### B. Image ingest path

Incoming pixels are unpacked into `pixels[]`, then binarized using:

- `pixel >= (1 << (INPUT_DATA_WIDTH - 1))`

The binarized bits are appended into `input_buffer`. `data_in_keep` determines which bytes are valid on each beat.

When the image is complete:

- `image_ready` pulses
- `input_bit_count` resets
- layer 1 receives a start pulse

#### C. Layer-to-layer chaining

There is a `layer_activations` array indexed by layer number:

- `layer_activations[0]` = input image bits
- `layer_activations[1]` = layer 1 outputs
- ...
- `layer_activations[LAYERS]` = output-layer scores

Each layer's `done` pulse is registered into `layer_start_pulse[l+1]`, which starts the next layer one cycle later.

Therefore the entire network acts like a strictly serialized pipeline:

- layer 1 must finish before layer 2 starts
- layer 2 must finish before layer 3 starts
- and so on

#### D. Output path

After the final layer finishes, the top level scans all output scores and selects the index with the largest signed 32-bit value.

That class index is driven onto:

- `data_out_data`
- `data_out_valid`
- `data_out_last`
- `data_out_keep`

The valid signal stays high until `data_out_ready` acknowledges the result.

## 4. Interconnection Summary

The module graph is:

```text
config_* stream
    |
    v
config_parser
    |
    +--> layer_memory[1] <--> compute_layer[1] --> layer_activations[1]
    |
    +--> layer_memory[2] <--> compute_layer[2] --> layer_activations[2]
    |
    +--> ...
    |
    +--> layer_memory[L] <--> compute_layer[L] --> layer_activations[L]

data_in_* stream
    |
    v
input unpack + binarize + input_buffer
    |
    v
layer_activations[0]
    |
    v
compute_layer[1] -> compute_layer[2] -> ... -> compute_layer[L]
    |
    v
argmax
    |
    v
data_out_* stream
```

At each layer boundary:

- activations move forward through `layer_activations`
- model parameters stay local to that layer's `layer_memory`
- control moves via one-cycle `start` and `done` pulses

Within each layer:

```text
compute_layer
    |  mem_layer_start / mem_read_weight / mem_read_thresh
    v
layer_memory
    |  rd_data_weights / rd_data_threshold
    v
compute_layer
    |  chunk x,w,last,threshold
    v
neural_processor
    |  valid_out / popcount_out
    v
compute_layer result store
```

## 5. Timing and Data Representation Details

### Activations

- Input image: one binarized bit per pixel
- Hidden-layer output: one bit per neuron
- Output-layer output: one signed 32-bit score per neuron

### Weight packing

- Weights are bit-packed per neuron
- Each neuron is padded to a byte boundary
- The memory read path reconstructs `CONFIG_BUS_WIDTH`-bit chunks from the packed stream

### Thresholds

- One 32-bit threshold per neuron
- Threshold payloads are stored as packed `CONFIG_BUS_WIDTH` words

### Layer execution granularity

For a layer with:

- `LAYER_INPUTS`
- `NUM_NEURONS`
- bus width `CONFIG_BUS_WIDTH`

the per-neuron chunk count is:

- `BYTES_PER_NEURON = ceil(LAYER_INPUTS / 8)`
- `BITS_PER_NEURON = BYTES_PER_NEURON * 8`
- `CHUNKS_PER_NEURON = ceil(BITS_PER_NEURON / CONFIG_BUS_WIDTH)`

The layer currently spends roughly:

- 1 cycle entering `PREFETCH`
- `CHUNKS_PER_NEURON` cycles in `COMPUTE_NEURON`
- 1 cycle in `FINISH_NEURON`

per neuron, not including top-level gaps between layers. That is a key reason the implementation is throughput-limited.

## 6. Current Architectural Constraints

These are important because they are more restrictive than the README alone suggests.

### Enforced or de facto fixed assumptions

- `PARALLEL_INPUTS` must equal the number of input elements per beat.
- `PARALLEL_INPUTS` is hard-failed to `8`.
- Every hidden layer's `PARALLEL_NEURONS[i]` must equal `PARALLEL_INPUTS`.
- `TOPOLOGY[0]` must be a multiple of `PARALLEL_INPUTS`.
- `config_parser` only really supports 32-bit or 64-bit config buses.
- Image ingest logic implicitly assumes one pixel per byte when counting `data_in_keep`.

### Functional simplifications

- Only one neuron is computed at a time in each layer.
- Only one `neural_processor` exists per `compute_layer`.
- Only one image is accepted at a time.
- No double-buffering exists for input images or intermediate activations.
- No overlap exists between layer execution stages.
- No overlap exists between model load and inference.

### Storage simplifications

- Weight memory is duplicated to support two-word aligned reads.
- Hidden-layer result vectors allocate `32 * NUM_NEURONS` bits even though only `NUM_NEURONS` bits are used.
- Configuration write strobes are not used to merge partial memory words.

### Interface/parameter drift

- `OUTPUT_DATA_WIDTH` is declared at the top level, but the final class index register is sized from `OUTPUT_BUS_WIDTH`.
- `data_out_keep` is driven as all-ones whenever `data_out_valid` is high, instead of being derived from an explicitly used output element width.

## 7. Main Performance Bottlenecks

If the redesign goal is "better performance", these are the dominant bottlenecks in the current RTL.

### 7.1 Neuron-level serialization

Each layer computes neurons one at a time. This is the biggest throughput limiter. Even if the input bus is wide, the design still reuses a single `neural_processor` for the whole layer.

### 7.2 Layer-level serialization

Layer `l+1` does not start until layer `l` has fully finished and written all results.

### 7.3 Full-image buffering before compute

The design must capture the entire image before layer 1 begins. There is no streaming overlap between image ingress and compute.

### 7.4 Single-image in flight

`data_in_ready` is tied to layer 1 being idle, so the next image cannot be accepted while the current one is being classified.

### 7.5 Per-neuron prefetch bubble

Each neuron incurs a synchronous-memory prefetch cycle before chunk processing begins.

### 7.6 Weight-memory duplication

The duplicated weight RAM solves an alignment problem, but increases memory cost and may complicate aggressive scaling.

## 8. Redesign-Relevant Guidance

A redesign agent should treat the current architecture as a correct-but-serial reference implementation, not as a high-performance baseline.

The clean architectural boundaries already present are:

- `config_parser` owns configuration-stream parsing
- `layer_memory` owns parameter storage and read alignment
- `compute_layer` owns layer scheduling and result collection
- `neural_processor` owns chunk-level BNN math
- `bnn_fcc` owns top-level orchestration, image ingress, and output formatting

Those boundaries make several redesign directions possible:

- replicate `neural_processor` to compute multiple neurons in parallel
- change `layer_memory` organization to store data in neuron-major, alignment-friendly banks
- stream activations directly into layer 1 instead of fully buffering the image
- pipeline adjacent layers or at least overlap image ingest with inference
- compress hidden-layer activation storage to true 1-bit arrays
- rework config loading so byte strobes are honored and bus widths are more generic

## 9. Bottom Line

The existing RTL is a straightforward reference implementation of a fully connected BNN:

- per-layer parameter memories
- sequential neuron traversal
- one XNOR-popcount accumulator per layer
- binary hidden activations
- score-based output layer plus final argmax

Its main value for a redesign effort is that the control/data decomposition is clear. Its main weakness is that almost all useful parallelism is currently left unused.
