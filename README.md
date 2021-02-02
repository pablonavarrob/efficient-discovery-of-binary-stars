# Efficient Discovery of Binary Stars: Master's Thesis for Lund University, 2020

Link to published work: https://lup.lub.lu.se/student-papers/search/publication/9012286

### Purpose

Even in the era of exponential increase in the amount of stellar data gathered, binaries are still often overlooked in observational data due to the special handling they require. The goal of this work is to develop a method capable of automatically and eciently identifying and extract double-lined spectroscopic binaries (SB2) from a spectroscopic survey, while being scalable and technically successful, and to identify and optimize the parameters that influence their detection.

### Method

We combine two state-of-the-art machine learning algorithms that group the spec- tra in the data-set in clusters based on their similarities, projecting them in a human readable manner (t-distributed Stochastic Neighbor Embedding, t-SNE), and automatically identify and retrieve those clusters that contain binary spectra (Density Based Spacial Clustering of Applica- tions with Noise, DBSCAN). These methods are then optimized for ecient recovery of binaries from a synthetic spectroscopic data-set, where we know exactly which stars are single and which are binaries.

### Results

We study the results following from 360 combinations of our method’s parameters and obtain a total average of recovered binaries of 57%. We show that under optimal conditions we are able to reach a recovery of 75%. We find that bluer spectral regions (450 nm - 600 nm) are better suited to identify binary stars than redder regions (600 nm - 900 nm) with our method. Not only this, but we also show that a moderate amount of noise can be beneficial and can improve the recovery of binary stars. Furthermore, we find that the stellar parameters that most influence the final recovery are the luminosity (or mass) ratio and the radial velocity di↵erent between the two stellar components of the binary system, while some standard stellar parameters can play a major role as well.

### Conclusions

We show that our method and the adopted combination of machine learning algorithms to be successful at automatically detect and retrieve binary stars from our synthetic spectroscopic data and we provide a list with guidelines for its application to real spectroscopic surveys.
