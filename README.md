# Masked Location Prediction using Transformers
Implementation of MOBERT - a BERT adaptation for human mobility modelling.
Part of the master thesis as partial fulfilment for the degree Master of Science in Engineering at DTU.
![cover map](figures/raw_data_world_map_cut.png)


### Directory structure:
------------

```
├── data-preprocessing-notebooks       <- Notebooks used for data preprocessing (section 4.2) visualisation 
├── figures                            <- All figures used in the thesis
├── model
│   ├── notebooks                      <- All scripts used for training and testing the model.
│   │   ├── entropy_plots.ipynb        <- Scripts used for plotting the entropy and predictability distribution (chapter 3).
│   │   ├── plot_test_results.ipynb    <- Scripts used for plotting test result figures (chapter 8)
│   │   ├── run_model.ipynb            <- Scripts used for training the model (chapter 7)
│   │   ├── test_model.ipynb           <- Scripts used for testing the model (chapter 8)
│   ├── saved_models                   <- All trained models (chapter 7 & 8)
│   ├── BERT.py                        <- All classes and functions to implement the BERT adaption MOBERT. 
│   ├── baseline.py                    <- The implementation of the baseline models (chapter 7 & 8)
│   ├── build_dataset.py               <- Class that extends torch.utils.data.Dataset class, contains functions connected to the creation of the dataset
│   ├── build_synthetic_data_frames.py <- Utilitiy funcitons for creating synthetic data sets
│   ├── evaluate_model.py              <- Utility Functions for Model Data Serialization and Deserialization
│   ├── fit_model.py                   <- Utility Functions and Model Training Loop for MOBERT
│   ├── mobility_entropy.py            <- Entropy Calculation Functions 
│   ├── test_model.py                  <- Test Loop Functions
├── output                             <- All training and test run results
├── README.md                          <- This README file
├── requirements.txt                   <- All packages installed in the virtual environment
├── thesis_david_hartel_s212588.txt    <- Thesis compiled as PDF

```

### Installing requirements
------------

    pip install -r requirements.txt

### Running the model
------------

Use `fit_model.py` to set parameters and train the model. 

### Thesis Abstract
------------

Understanding, modeling, and predicting human mobility in urban areas is an essential task for various domains such as transportation modelling, disaster risk management, and infectious disease spreading. In this thesis, we introduce a custom BERT­based model, MOBERT, designed for human mobility modelling. Trained and tested on a pro­ cessed dataset derived from the Copenhagen Network Study, which captures location data from smartphones of 840 individuals, MOBERT effectively predicts masked loca­ tions, outperforming three baseline models and achieving superior results compared to existing­next location prediction approaches. Our analysis of the impact of additional features, including user ID, time, and location rank, on prediction accuracy showed no significant improvements. Furthermore, we evaluated the model’s performance across different grid sizes and location types, emphasizing its proficiency in learning individual­ specific regular patterns while highlighting challenges with explorative mobility patterns. Further limitations include the difficulty in comparing our results with other studies due to the different datasets used across studies and our bidirectional approach, which dif­ fers from the one­directional next location prediction task. We suggest future research focus on further parameter optimization, employing more comprehensive datasets, and fine­tuning to a next location prediction task. Furthermore, enhancing privacy while main­ taining utility remains a critical area for future exploration.



