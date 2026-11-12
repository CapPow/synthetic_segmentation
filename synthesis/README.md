# Synthetic Herbarium Specimen Generation Pipeline

Procedural pipeline used to generate the LaForge10k synthetic training set. Included to allow inspection of the synthesis method. This is not packaged as a turnkey reproduction environment; running it would require modifying hardcoded paths in the scene file and reconciling the addon with its original upstream source.

## Pipeline Overview

1. Opens a Blender scene (`light_box.blend`) containing a herbarium sheet, lighting rig, camera, and compositor configured to output both RGB and a pass-index-based segmentation mask.
2. Randomizes scene properties: paper yellowing, lighting color/energy, camera distance, presence and placement of reference props (color check cards, scale bars, metal washers).
3. Uses the [tree-gen](https://github.com/friggog/tree-gen) parametric tree addon to generate a 3D plant structure with randomized branch topology, leaf count, leaf shape, and bark/leaf material assignment.
4. Applies a linear Z-scale ("pressing") to flatten the specimen onto the herbarium sheet plane, positions it within the paper bounds, and assigns a randomly selected pre-generated label texture.
5. Optionally generates additional specimens per sheet using remixed parameters from the first.
6. Splits joined geometry by material for pass-index annotation (leaf vs. woody tissue).
7. Renders RGB image + semantic mask via the compositor File Output node; writes a JSON sidecar recording generation parameters for each sample.

## Contents

| Path | Description |
|------|-------------|
| `specimen_generator.py` | Main script. All scene manipulation, parameter randomization, tree-gen invocation, positioning, and output naming. Run via Blender's scripting interface or headless CLI. |
| `light_box.blend` | Scene file (~270 MB). Contains the full herbarium "set": sheet mesh, lighting array, camera, compositor node tree (RGB + pass-index mask output), all materials (16 bark types, 14 leaf types, label materials, prop materials), and reference geometry. Output directory paths for renders and the relative path to label images are hardcoded here and would need editing for a different environment. |
| `label_generation/output/` | 10,000 pre-rendered herbarium label images used as decal textures on a plane object during synthesis. |
| `assets/textures/` | Standalone copies of leaf surface textures used in the scene materials, provided for quick visual inspection without opening the `.blend` file. Each leaf material is built from a rectangular plane with two paired faces: the adaxial (upper) surface and the abaxial (lower) surface, distinguished by an `_a` suffix (e.g., `cordate_texture_1.png` + `cordate_texture_1_a.png`). These same textures are also baked into `light_box.blend`. |
| `.gitattributes` | Git LFS tracking for binary files exceeding GitHub's 100 MB threshold. |

## Dependencies and Caveats

- **Blender**: The script was developed and run against an older Blender release (4.x). It uses `bpy.data.batch_remove`, `bpy.ops.object.select_pattern`, and other APIs that may have changed or been deprecated in subsequent versions.
- **tree-gen addon** ([upstream](https://github.com/friggog/tree-gen)): Required. The script monkeypatches two functions at runtime (`Tree.create_leaf_mesh` for leaf orientation constraints, `Tree.points_for_floor_split` to prevent infinite loops under degenerate parameters). Additionally, the local copy of the addon may need a version bump for the running blender version.
- **Hardcoded paths**: The `.blend` file references label images at a relative path (`.//label_generation/output/`) and render outputs at an absolute directory set in the compositor File Output node. Both must be adjusted before successful execution on a different system.

## Citation

The synthesis method is described in full in the associated manuscript. This code is provided as supplementary material for inspection.
