# Third-party notices

The source code in this repository is released under the [MIT License](LICENSE). That license
covers the code only. The items below belong to their respective owners and keep their own
terms.

## tf_flowers dataset

- **Source:** the TensorFlow team's
  [`flower_photos`](https://www.tensorflow.org/datasets/catalog/tf_flowers) archive, loaded
  through TensorFlow Datasets.
- **License:** every photo is licensed by its photographer under
  [Creative Commons Attribution 2.0 (CC BY 2.0)](https://creativecommons.org/licenses/by/2.0/).
  The full list of photographers is in the `LICENSE.txt` file inside the archive.

The dataset is **not** included in this repository; it is downloaded by TensorFlow Datasets the
first time you train or evaluate.

### Photos included in this repository

Thirteen photos from the dataset's test split appear here, unmodified apart from resizing in
the figures. The demo examples are the first correctly classified test photo of each class.

| Used in | File in the dataset | Photographer (CC BY 2.0) |
|---|---|---|
| `demo/examples/daisy.jpg` | `daisy/8708143485_38d084ac8c_n.jpg` | [Andrés Moreira](https://www.flickr.com/photos/andrix/8708143485/) |
| `demo/examples/dandelion.jpg` | `dandelion/2693136371_dde2570813.jpg` | [Andrey](https://www.flickr.com/photos/akras/2693136371/) |
| `demo/examples/roses.jpg` | `roses/488849503_63a290a8c2_m.jpg` | [Olga Khomitsevich](https://www.flickr.com/photos/zemzina/488849503/) |
| `demo/examples/sunflowers.jpg` | `sunflowers/8174935717_d19367d502.jpg` | [Stephane Mignon](https://www.flickr.com/photos/topsteph53/8174935717/) |
| `demo/examples/tulips.jpg` | `tulips/8904780994_8867d64155_n.jpg` | [jsogo](https://www.flickr.com/photos/jsogo/8904780994/) |
| `reports/figures/confident_mistakes.png` | `roses/7345657862_689366e79a.jpg` | [Rosa Dik 009 -- on & off](https://www.flickr.com/photos/18_2rosadik36/7345657862/) |
| `reports/figures/confident_mistakes.png` | `tulips/15082212714_ff87e8fcb1_m.jpg` | [Swallowtail Garden Seeds](https://www.flickr.com/photos/swallowtailgardenseeds/15082212714/) |
| `reports/figures/confident_mistakes.png` | `roses/3713368809_eba7fa2fbf_m.jpg` | [Antony Stanley](https://www.flickr.com/photos/antonystanley/3713368809/) |
| `reports/figures/sample_predictions.png` | `tulips/4575963749_2418ff8768.jpg` | [Perry McKenna](https://www.flickr.com/photos/63567936@N00/4575963749/) |
| `reports/figures/sample_predictions.png` | `tulips/8762193202_0fbf2f6a81.jpg` | [bark](https://www.flickr.com/photos/barkbud/8762193202/) |
| `reports/figures/sample_predictions.png` | `tulips/13903988248_22da33f341.jpg` | [Blondinrikard Fröberg](https://www.flickr.com/photos/blondinrikard/13903988248/) |
| `reports/figures/sample_predictions.png` | `dandelion/4635296297_9ce69e4a6e.jpg` | [Kasya Shahovskaya](https://www.flickr.com/photos/kasya/4635296297/) |
| `reports/figures/sample_predictions.png` | `tulips/8712268519_f4c2c39a06_n.jpg` | [PS-OV-ART Patty Sue O'Hair-Vicknair, Artist](https://www.flickr.com/photos/psovart/8712268519/) |

The demo examples also appear in `reports/figures/demo.gif`.

If you hold rights to any of these photos and want them removed, please
[open an issue](https://github.com/esammostafa9-cloud/floravision-flower-classification/issues/new)
and they will be removed promptly.

## Pretrained weights

The model is initialised from Keras' ImageNet-pretrained ResNet-50 weights, which are ported from
the weights released by Kaiming He under the MIT License
([source](https://github.com/fchollet/deep-learning-models#licensing)). The ImageNet dataset has
its own [terms of access](https://image-net.org/download).

The trained model is published as a GitHub release asset, not stored in the repository. Because
it starts from ImageNet-pretrained weights, it is offered for non-commercial research and
educational use.

## Software dependencies

Installed separately via `pip`; none are bundled here.

| Package | License |
|---|---|
| TensorFlow, Keras, TensorFlow Datasets | Apache-2.0 |
| NumPy | BSD-3-Clause (with bundled MIT, Zlib, 0BSD and CC0 components) |
| pandas, scikit-learn | BSD-3-Clause |
| Matplotlib | Matplotlib license (PSF-based) |
| Pillow | MIT-CMU |
| PyYAML | MIT |
| Gradio (demo only) | Apache-2.0 |
