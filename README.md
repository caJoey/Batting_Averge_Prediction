# 570_Project

# Project Deliverables:

## 1. Paper

Paper is in Paper.pdf

## 2. Presentation

Presentation is in Final Presentation Slides.pdf, and video is linked [here](https://youtu.be/mGYSYoILyjo).

# How to run the code

The demo can be found [here](https://youtu.be/rNpo8EM49rs)

## Which files to run

1. adv_models.py
 - This is the main Python file that prints out MAE for our models
 - Test and Train sets can be adjusted using TRAIN_MAX and TEST_MAX variables in the code (i.e. for a test set that is 2023-2025, set TRAIN_MAX to 2022 and TEST_MAX to 2025).
 - Then just run the file and it will print out MAE results
2. Steamer 2025 results
 - One comparison in the paper we did was with Steamer. This result can be recreated by going to Villain_Projections folder and running villain_projections.py; this prints out the 2025 Steamer MAE for the same batters tested on in adv_models.py
3. The Bat X 2024 results
 - The other comparison in the paper we did was with The Bat X. This result can be recreated by going to Villain_Projections folder and running batx_villain_projections.py; this prints out the 2024 The Bat X MAE for the same batters tested on in adv_models.py

## Other files

Other folders and files can be ignored as these were used for web scraping, experimenting, or organiziing data.
