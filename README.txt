Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


This is the code for the team "DIKU-IR" in the Spotify sequential skip prediction challenge (https://www.crowdai.org/challenges/spotify-sequential-skip-prediction-challenge)

This project consists of two python files, and two folders sat up with example data to run on. The easiest way to run the code is to download the whole project and run the code from inside the top level folder.

data_preprocessing.py contains all the necessary code for processing the data. In the folder /data/ there are small examples of track data and session data as provided in the competition, so the preprocessing code can run as is. There is no small example file for the test data.
The processing depends on a small dictionary in the folder cat_dict, which translates the categorical string variables to integers for later one hot encoding.

The file tf_network.py contains the implementation of our model, and the necessary code for both training the network, and running the trained model on a trained model.