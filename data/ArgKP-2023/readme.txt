	NAME: IBM Debater(R) - ArgKP-2023

VERSION: v1

RELEASE DATE: December 5, 2023

DATASET OVERVIEW

9,281 (argument, key point) pairs labeled as matching/non-matching/undecided, for 10 controversial topics. 
For each pair, the topic and stance are also indicated.

The dataset is released under the following licensing and copyright terms:
• (c) Copyright IBM 2023. Released under Community Data License Agreement – Sharing, Version 1.0 (https://cdla.io/sharing-1-0/).

The dataset is described in the following publication (referred to as the "ArgKP-Large" test set): 

• Welcome to the Real World: Efficient, Incremental and Scalable Key Point Analysis. Lilach Eden, Yoav Kantor, Matan Orbach, Yoav Katz, Noam Slonim, Roy Bar-Haim. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track. 2023


Please cite this paper if you use the dataset.

CONTENTS

The CSV file, ArgKP-2023_dataset.csv, contains the following columns for each (argument, key point) pair:
1. topic
2. argument
3. key_point	
4. stance: 1 (pro) / -1 (con)
5. label: 1 (matching)/ 0 (non-matching)/ -1 (undecided)

Note: In the experiments described in the paper, the undecided examples were excluded.