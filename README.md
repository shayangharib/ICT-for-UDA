# Extending Interpolation Consistency Training for Unsupervised Domain Adaptation

This is the repository of the implementtion for the presented method in our paper: 
"Extending Interpolation Consistency Training for Unsupervised Domain Adaptation" 
by [Shayan gharib](https://scholar.google.com/citations?user=neb2vi0AAAAJ&hl=en) and [Arto Klami](https://scholar.google.com/citations?user=v8PeLGgAAAAJ&hl=en). 

Our paper is accepted for publication at the [International Joint Conference on Neural Networks (IJCNN) 2023](https://2023.ijcnn.org/). 

---

To run an experiment, use the file ``main.py``:

python main.py --index [experiment_index]

To change the settings of each experiment, use the setting yaml file: ``settings.yml``.

To run the experiment for MNIST --> MNIST-M setup, the MNIST-M dataset needs to be manually downloaded from [this link](https://drive.google.com/file/d/0B_tExHiYS-0veklUZHFYT19KYjg/view?resourcekey=0-DE7ivVVSn8PwP9fdHvSUfA), and be placed in "datasets/MNISTM" directory. The other two datasets (i.e. MNIST and USPS) are automatically downloaded
and processed.
