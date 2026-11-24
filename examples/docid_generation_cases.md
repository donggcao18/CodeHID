# Qualitative Examples of Hierarchical DocID Generation

This page presents three real examples from the CoSQA development split. We retain the complete target code snippets to make the correspondence between natural-language queries, code implementations, and hierarchical DocIDs directly inspectable. These examples are qualitative illustrations of the generation process rather than an additional experiment.

Hierarchical DocIDs are assigned to code snippets during offline indexing and remain static and query-independent. During inference, CodeHID does not create new identifiers; it autoregressively predicts paths within the existing DocID space under constrained decoding. The listed DocIDs are the final generated outputs returned by constrained decoding.

**Offline indexing:**  
code snippets → static hierarchical DocIDs

**Online retrieval:**  
query → constrained DocID generation → indexed code

Each DocID contains four learned codebook tokens, serialized as `a...b...c...d...`. The tokens are not manually assigned semantic labels. A shared prefix only indicates that code snippets occupy the same learned region at the corresponding level of the hierarchy.

## Case 1: Successful Navigation to the Target Address

**Query**

> python split strings into list of lines

**Target code**

```python
def split_multiline(value):
    """Split a multiline string into a list, excluding blank lines."""
    return [element for element in (line.strip() for line in value.split('\n'))
            if element]
```

**Target DocID:** `a135b75c249d248`

The target path can be read from coarse to fine as:

```text
a135
  -> a135b75
      -> a135b75c249
          -> a135b75c249d248
```

The top five generated DocID candidates returned by constrained decoding are:

| Rank | Generated DocID | Relation to the target path |
| ---: | --- | --- |
| 1 | `a135b75c249d248` | Exact target |
| 2 | `a135b75c202d198` | Shares the first two levels |
| 3 | `a135b75c249d188` | Shares the first three levels |
| 4 | `a135b75c231d212` | Shares the first two levels |
| 5 | `a135b75c247d26` | Shares the first two levels |

**Observation.** The model reaches the exact static address at rank 1. All five predictions retain the two-level prefix `a135b75`, while the deeper tokens provide finer-grained separation among candidates within that learned region.

## Case 2: Coarse Region Localization with Fine-Grained Ambiguity

**Query**

> loading a series of images in python and resizing them

**Target code**

```python
def load_preprocess_images(image_paths: List[str], image_size: tuple) -> List[np.ndarray]:
    """
    Load and pre-process the images specified with absolute paths.

    :param image_paths: List of images specified with paths.
    :param image_size: Tuple to resize the image to (Channels, Height, Width)
    :return: A list of loaded images (numpy arrays).
    """
    image_size = image_size[1:]  # we do not need the number of channels
    images = []
    for image_path in image_paths:
        images.append(load_preprocess_image(image_path, image_size))
    return images
```

**Target DocID:** `a165b109c217d160`

The top five generated DocID candidates returned by constrained decoding are:

| Rank | Generated DocID | Relation to the target path |
| ---: | --- | --- |
| 1 | `a165b109c11d91` | Shares the first two levels |
| 2 | `a165b109c11d193` | Shares the first two levels |
| 3 | `a165b109c217d50` | Shares the first three levels |
| 4 | `a165b109c217d160` | Exact target |
| 5 | `a165b109c151d47` | Shares the first two levels |

**Observation.** The target occurs at rank 4. Every displayed prediction shares the prefix `a165b109`, while the alternatives branch at the third or fourth level. The hierarchy therefore narrows generation to a shared learned region, while resolving target-level ambiguity remains the key fine-grained retrieval challenge.

## Case 3: Local Competition within a Shared Prefix Region

**Query**

> python subplot set the whole title

**Target code**

```python
def set_title(self, title, **kwargs):
    """Sets the title on the underlying matplotlib AxesSubplot."""
    ax = self.get_axes()
    ax.set_title(title, **kwargs)
```

**Target DocID:** `a165b137c232d246`

The top five generated DocID candidates returned by constrained decoding are:

| Rank | Generated DocID | Relation to the target path |
| ---: | --- | --- |
| 1 | `a165b137c232d34` | Shares the first three levels |
| 2 | `a165b137c232d246` | Exact target |
| 3 | `a165b137c232d37` | Shares the first three levels |
| 4 | `a165b137c232d227` | Shares the first three levels |
| 5 | `a165b137c238d61` | Shares the first two levels |

**Observation.** The target occurs at rank 2. The first four predictions share the three-level prefix `a165b137c232` and compete only at the final codebook token. In this example, the remaining competition is localized to the finest level of the hierarchy.

## Takeaway

Together, the examples illustrate three typical behaviors of hierarchical DocID generation: a successful target navigation, branching within a shared coarse prefix, and local competition among candidates sharing a deeper prefix. The numeric tokens should be interpreted as learned addresses and prefix relationships, not as human-readable category names.
