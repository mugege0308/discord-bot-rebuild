# ComfyUI workflow files

Place your ComfyUI API-format workflow JSON here.

Default file name:

```text
comfy_workflow.json
```

The bot reads the path from:

```env
COMFY_WORKFLOW_FILE=workflows/comfy_workflow.json
```

The workflow must include:

- a positive `CLIPTextEncode` node
- a negative `CLIPTextEncode` node
- a sampler with `positive` / `negative` inputs
- a `RandomNoise` node or another node with a `seed` input
- a `SaveImage` node

Normally, node IDs are detected automatically. The bot follows the sampler's `positive` and `negative` links to find `CLIPTextEncode`, detects `RandomNoise.noise_seed` or a `seed` input, and checks for `SaveImage`.

Only set these IDs in `.env` if automatic detection does not match your workflow:

```env
COMFY_POSITIVE_NODE_ID=
COMFY_NEGATIVE_NODE_ID=
COMFY_SEED_NODE_ID=
```

## Animagine XL 4.0 base-only workflow

`comfy_workflow.json` is set up as a simple Animagine XL 4.0 txt2img workflow.

Current shape:

- `CheckpointLoaderSimple`
- positive `CLIPTextEncode`
- negative `CLIPTextEncode`
- `EmptyLatentImage` at `1024x1024`
- one base `KSampler`
- `VAEDecode`
- `SaveImage`

Refiner nodes are not used. Animagine XL is an SDXL base model, so this workflow keeps generation in the base sampler only.

Default sampling settings:

- steps: `30`
- cfg: `6`
- sampler: `euler_ancestral`
- scheduler: `normal`
- denoise: `1`

The bot replaces only the Positive / Negative prompt text and randomizes the sampler seed before queueing the workflow.

The previous workflow was backed up as:

```text
workflows/comfy_workflow.backup_before_base_only.json
```
