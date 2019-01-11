Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


This is the teams DIKU-IR submission for the spotify sequential skip prediction challenge.

The file data_processing.py contains all the necesarry code for processing the data given at the competition, and and folder structure where the full dataset can be copied into.
Currenty there is small "dummy" files representing the train data and track data so the code is runable.

The file tf_generator.py contains implementation of our model, and the necesarry code to both train the model and use it for prediction.