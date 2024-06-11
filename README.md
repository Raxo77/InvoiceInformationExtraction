## Invoice Information Extraction

This repository contains code, sample data and (aggregated) results of a Bachelor's thesis on *Improving Invoice
Information Extraction: An Investigation into Ensemble Learning Methods in Named Entity Recognition*.  
Specifically, the thesis investigates whether a combination of approaches both specific to the invoice extraction domain
and of analogous domains in the form of an ensemble results in increased performance of the extraction system.
Entities targetted for extraction are:

* Invoice Date
* Invoice Number
* Invoice Gross Amount
* Invoice Tax Amount
* Order Number
* Issuer Name
* Issuer IBAN
* Issuer Address

The dataset used throughout the thesis is the [Inv3D](https://felixhertlein.github.io/inv3d/) dataset containing
information on 25,000 synthetically generated English invoices across 100 different layouts.

---

## Approaches and Algorithms employed

In total, six different models are employed first in isolation and then in concert. Respective model and ensemble
performance scores are then compared to assess whether the ensemble structure results in increased (or descreased)
performance. Each submodel is grounded in prior work with its architecture and conceptuality briefly discussed below.

#### Submodel 1
#### Submodel 2
#### Submodel 3
#### Submodel 4
#### Submodel 5
#### Submodel 6


---

## Results

---

## Folder Structure

---

## Citations

@article{Hertlein2023,  
title = {Inv3D: a high-resolution 3D invoice dataset for template-guided single-image document unwarping},  
author = {Hertlein, Felix and Naumann, Alexander and Philipp, Patrick},  
year = 2023,  
month = {Apr},  
day = 29,  
journal = {International Journal on Document Analysis and Recognition (IJDAR)},  
doi = {10.1007/s10032-023-00434-x},  
ISSN = {1433-2825},  
url = {https://doi.org/10.1007/s10032-023-00434-x}  
