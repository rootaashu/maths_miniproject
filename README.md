# Maths Mini Project — Matrix-Based Image Processing

<p align="center">
  <strong>Understanding Mathematics Through Image Processing</strong>
</p>

A Python-based image processing toolkit that explores the practical applications of linear algebra, matrix operations, and numerical methods in digital image manipulation.

The project implements mathematical operations such as matrix multiplication, transposition, determinants, matrix inversion, affine transformations, convolution, and Singular Value Decomposition (SVD) to perform various image-processing operations.

Rather than treating image processing as a black box, this project explores the mathematics behind how images are transformed, filtered, and reconstructed.

---

## Overview

Digital images can be represented as matrices of numerical values. By applying mathematical operations to these matrices, we can manipulate images in different ways.

This project demonstrates how concepts from linear algebra and numerical computing can be used to build an image-processing system using Python and NumPy.

### Key Features

- Custom matrix operations and linear algebra implementations.
- Geometric image transformations using affine matrices.
- Image filtering through convolution kernels.
- Colour manipulation and grayscale conversion.
- Image compression using Singular Value Decomposition.
- Nearest-neighbour and bilinear interpolation.
- Command-line interface for running image-processing demonstrations.

---

## Project Modules

### 1. Matrix Operations

Implemented fundamental matrix operations used throughout the project.

- Identity matrix generation.
- Matrix transposition.
- Matrix multiplication.
- Determinant calculation using Gaussian elimination.
- Matrix inversion using Gauss-Jordan elimination with pivoting.

These operations form the mathematical foundation of the image transformations.

### 2. Geometric Transformations

Uses homogeneous coordinates and affine transformation matrices to manipulate image geometry.

Supported transformations:

| Transformation | Description |
|---|---|
| Translation | Moves an image along the X and Y axes |
| Rotation | Rotates an image by a specified angle |
| Scaling | Resizes an image |
| Shearing | Slants the image along an axis |
| Reflection | Flips an image horizontally or vertically |
| Composition | Combines multiple transformations |

Interpolation methods:
- Nearest-neighbour interpolation.
- Bilinear interpolation.

### 3. Image Filtering

Implements image filters using convolution kernels.

Available filters include:

- Gaussian Blur
- Sharpen
- Edge Detection
- Emboss
- Sobel X
- Sobel Y
- Sobel Magnitude

These operations demonstrate how kernel matrices can extract features and modify image characteristics.

### 4. Colour Operations

Provides mathematical manipulation of image colour channels.

Supported operations:

- Grayscale conversion.
- Sepia effect.
- Red and blue channel swapping.
- Negative transformation.
- Brightness adjustment.
- Contrast adjustment.

### 5. SVD Image Compression

Uses Singular Value Decomposition to approximate image matrices with lower-rank representations.

The compression process reduces the number of singular values used to reconstruct an image.

Different values of `k` can be used to compare image quality and compression.

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Matrix and numerical operations |
| Matplotlib / image libraries | Image visualisation and handling, where used |
| Git | Version control |

---

## Project Structure

```text
maths_miniproject/
│
├── main.py
├── requirements.txt
├── sample.png
│
├── matrix_image/
│   ├── __init__.py
│   ├── matrix_ops.py
│   ├── transforms.py
│   ├── filters.py
│   ├── color.py
│   ├── svd_compress.py
│   ├── io_utils.py
│   └── demo.py
│
├── outputs/
│
├── .gitignore
└── README.md
```

---

## Installation and Setup

### Prerequisites

- Python 3.10 or later recommended.
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/rootaashu/maths_miniproject.git
cd maths_miniproject
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Linux / EndeavourOS / Arch Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```powershell
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Running the Project

Place your input image in the project directory.

For example:

```text
maths_miniproject/
└── sample.png
```

Run the demonstration:

```bash
python main.py demo sample.png
```

The demonstration executes the available image-processing operations and generates output figures.

Generated figures are saved in:

```text
outputs/
```

---

## Mathematical Concepts Explored

This project connects several mathematical concepts with practical image processing.

| Mathematical Concept | Application |
|---|---|
| Matrix multiplication | Combining transformations |
| Matrix inverse | Inverse mapping of image coordinates |
| Homogeneous coordinates | Representing affine transformations |
| Linear algebra | Image representation and manipulation |
| Convolution | Image filtering and edge detection |
| Interpolation | Pixel reconstruction |
| Singular Value Decomposition | Image compression |
| Gaussian elimination | Determinants and matrix inversion |

---

## Learning Outcomes

Through this project, the following concepts are explored practically:

- Understanding how digital images can be represented as matrices.
- Implementing matrix operations without relying entirely on built-in matrix multiplication.
- Applying affine transformations using homogeneous coordinates.
- Understanding inverse mapping in geometric image transformations.
- Exploring convolution kernels and their applications.
- Understanding the relationship between linear algebra and computer vision.
- Applying SVD to image compression and reconstruction.
- Organising mathematical implementations into reusable Python modules.

---

## Future Improvements

Potential extensions to the project include:

- Interactive image-processing interface.
- Real-time image transformations.
- Additional convolution filters and edge-detection algorithms.
- Comparison of different interpolation techniques.
- Compression quality analysis using PSNR and reconstruction error.
- Performance comparison between custom matrix operations and NumPy implementations.
- Support for batch image processing.

---


