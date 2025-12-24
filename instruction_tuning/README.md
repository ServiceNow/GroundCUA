# GroundCUA Instruction Tuning Dataset

The instruction tuning dataset used to train GroundNext models can be found in the [GroundCUA Hugging Face repository](https://huggingface.co/datasets/ServiceNow/GroundCUA/tree/main) under the name `instruction_tuning.tar.gz`. 

To extract the dataset:
```bash
tar -xzvf instruction_tuning.tar.gz
```

---

## 📁 Dataset Files

To create the dataset, we first perform deduplication of elements for every platform using the label and icon crop (using pHash). This gives us around 900K–1M elements, which we then use to build instructions through prompting and heuristic methods as described in the paper.

### Direct Instructions
Straightforward instructions that directly describe UI elements based on different attributes:

| File | Description |
|------|-------------|
| `direct_description_instructions.json` | Detailed descriptions of UI elements including appearance, color, position, and function |
| `direct_general_templates_instructions.json` | Template-based instructions where element name is inserted |
| `direct_text_instructions.json` | Instructions referencing the text/label of UI elements |
| `direct_icon_instructions.json` | Instructions describing icon-based UI elements |
| `direct_miscellaneous_instructions.json` | Other direct instruction types not covered above |

### Functional Instructions
Instructions that describe UI elements by their purpose or action:

| File | Description |
|------|-------------|
| `functional_instructions.json` | Instructions describing what the UI element does (e.g., "Save your work") |
| `functional_instructions_extra.json` | Initial set of functional instructions created for the smallest elements of each platform, which provided gains in early experiments |

### Spatial Instructions

| File | Description |
|------|-------------|
| `spatial_data.json` | Instructions that use spatial relationships to describe element locations |

### Other

| File | Description |
|------|-------------|
| `remaining_instructions.json` | Remaining data not used to train models. These are **not** low quality—they simply did not fit within our 700K training budget |

---

## 📄 Data Format

Most files follow this JSON structure:

```json
{
    "id": "196be5155723e62a6b4b9479a443c6334ade2526026d4bb4730b3f3d217515aa",
    "bbox": [187.97, 917.10, 223.86, 952.10],
    "text": "export document",
    "category": "Button",
    "platform": "Veusz",
    "all_instructions": {
        "description": "The export document icon is located in the bottom toolbar, third from the left...",
        "functional_instruction": "Save your work to a file for future use or sharing.",
        "simple_instruction": "Go to `export document`."
    },
    "instruction": "The export document icon is located in the bottom toolbar..."
}
```

**Note:** `spatial_data.json` and `functional_instructions_extra.json` use a different format:

```json
{
    "id": "5a192612c5c2c4b982077c0b80fb6732931819a5180bb3c449fc323f7d12384b",
    "platform": "Affine",
    "coordinate": [169.41, 153.08],
    "instruction": "Select the 'Demo Workspace' to access its contents and settings"
}
```

These files do not include bounding boxes, only center coordinates. You can retrieve the bounding box using the `find_element_for_instruction` method in `utils.py`.

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Unique identifier used to locate the image and annotation file in the GroundCUA dataset |
| `bbox` | `[x1, y1, x2, y2]` | Bounding box coordinates of the UI element |
| `text` | `string` | Human-annotated label or description of the element |
| `category` | `string` | UI element type (e.g., Button, Menu, Navigation) |
| `platform` | `string` | Source application (e.g., Veusz, VSCode, Zotero) |
| `all_instructions` | `object` | Dictionary containing all instruction variants (multiple instructions generated per element) |
| `instruction` | `string` | The selected instruction used to train GroundNext models |

---

**Note:** For training GroundNext, we use all data files except `remaining_instructions.json`, totaling around 699K instructions. These files contain the `instruction` key which we use to create the training dataset. The `remaining_instructions.json` file contains an additional ~450K elements with instructions that were not used for training.

## Linking to GroundCUA Assets

Use the `id` and `platform` fields to locate the corresponding files in the main GroundCUA repository:

| Asset | Path |
|-------|------|
| Annotations | `GroundCUA/data/{platform}/{id}.json` |
| Images | `GroundCUA/images/{platform}/{id}.png` |

## Additional Notes & Tips

### Filtering Criteria Used

- **Element size threshold:** We filtered out most elements with a relative size above 0.005 (element area / image area). This removed only a small fraction of the data but provided good gains, as larger elements are often already well-known to the model. Additionally, larger bounding boxes have more valid click points, but since we only fit a single point during SFT, this can lead to unnecessary penalization.

- **Text length threshold:** We set a maximum text length of 30 characters. Longer text typically corresponded to code elements or heavy-text UI components, which were less useful for training.

- **Text element size exception:** For elements containing text, we relaxed the size threshold to allow relative sizes up to 0.1. We did not extensively ablate these decisions though so these are just rough guidelines.

### Future Experimentation Ideas

- **Elements per screenshot:** The current dataset does not cap the number of elements per screenshot, meaning some dense screenshots contribute many more training examples than sparse ones. Early experiments with capping elements per screenshot showed promising results. Since elements are randomly selected, they are roughly sampled proportionally to density, but this could be worth experimenting with further.

- **Dataset mixing:** Beyond 700K examples, we highly recommend mixing GroundCUA with other datasets such as **UGround**, **GTA-1**, and **ScaleCUA**—especially for mobile and web domains. Our experiments showed improved generalization on mobile and web with just 5% data from UGround. We strongly encourage combining GroundCUA with complementary datasets for best results.

- **Using Remaining Data:** We never got time to experiment with the remaining data. This is something that could provide gains beyond the 700k data points especially for icons on Desktop platforms.
