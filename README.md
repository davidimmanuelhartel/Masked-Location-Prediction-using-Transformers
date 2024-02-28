# Masked Location Prediction using Transformers
Implementation of MOBERT - a BERT adaptation for human mobility modelling.

![cover map](figures/raw_data_world_map_cut.png)


Abstract

Understanding, modeling, and predicting human mobility in urban areas is an essential task for various domains such as transportation modelling, disaster risk management, and infectious disease spreading. In this thesis, we introduce a custom BERT­based model, MOBERT, designed for human mobility modelling. Trained and tested on a pro­ cessed dataset derived from the Copenhagen Network Study, which captures location data from smartphones of 840 individuals, MOBERT effectively predicts masked loca­ tions, outperforming three baseline models and achieving superior results compared to existing­next location prediction approaches. Our analysis of the impact of additional features, including user ID, time, and location rank, on prediction accuracy showed no significant improvements. Furthermore, we evaluated the model’s performance across different grid sizes and location types, emphasizing its proficiency in learning individual­ specific regular patterns while highlighting challenges with explorative mobility patterns. Further limitations include the difficulty in comparing our results with other studies due to the different datasets used across studies and our bidirectional approach, which dif­ fers from the one­directional next location prediction task. We suggest future research focus on further parameter optimization, employing more comprehensive datasets, and fine­tuning to a next location prediction task. Furthermore, enhancing privacy while main­ taining utility remains a critical area for future exploration.



