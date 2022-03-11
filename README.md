# TROMPA-MER: an open data set for personalized Music Emotion Recognition

Source code to reproduce experiments and visualisation platform presented to IEEE Transactions on Affective Computing 2021.

-- Juan Sebastián Gómez-Cañón, Nicolás Gutiérrez-Páez, Lorenzo Porcaro, Alastair Porter, Estefanía Cano, Perfecto Herrera, Aggelos Gkiokas, Patricia Santos, Davinia Hernández-Leo, Casper Karreman, and Emilia Gómez

## Abstract

We present a data set for the analysis of personalized Music Emotion Recognition (MER) systems.
We developed the Music Enthusiasts platform aiming to improve the gathering and analysis of the so-called “ground truth” needed as input to such systems.
Our platform involves citizen science strategies to engage with audiences and generate large-scale music emotion annotations -- the platform presents didactic information and musical recommendations as incentivization, while collecting data regarding demographics, language, and mood from each participant.
Participants annotated each music excerpt with single free-text emotion word (in native language), forced-choice of distinct emotion categories, preference, and familiarity.
Additionally, participants stated the reasons for each annotation -- including those particular to emotion perception and emotion induction.
An extensive analysis of the participants' ratings is presented and a novel personalization strategy is evaluated.
Results evidence that using the “collective judgement” as prior knowledge for active learning allows for more effective personalization of MER systems for this particular data set.
Our data set is publicly available and we invite researchers to use it for testing MER systems.

## Online Website

With UPF-VPN: http://mirlab-web1.s.upf.edu:8050/vis-mtg-mer

External: https://trompa-mtg.upf.edu/vis-mtg-mer/

## Reproduce results

#### To run with a virtual environment:

Create the virtual environment:

```
python3 -m venv trompa-venv
source trompa-venv/bin/activate
pip3 install -r requirements.txt
```

Copy the `sklearn.py` into the virtual environment library:

```
cp xgboost/sklearn.py trompa-venv/lib/python3.8/site-packages/xgboost
```

#### To merge annotations with playlist summary:

```
python3 merge_research_data.py
```

#### To obtain generalized word shift graphs from textual data:

We use Shannon's entropy to produce the shift graphs in the paper.

```
python3 nlp_shifterator.py -shift [proportion [prop], Shannon-Entropy [shan], Jensen-Shannon Divergence [jsd]]
```

#### To reproduce personalization results:

Contact juansebastian.gomez[at]upf[dot]edu to receive the processed feature set. We use only data from users that annotated more than 80 excerpts, and perform 10 iterations with 5 queries each.

```
python3 evaluate_dataset.py -n 80 -e 10 -q 5
```

The previous step will train all models for all users - it will take approximately 2 hours. After the process is complete, you need to analyze the results:

```
python3 analysis.py
```

This will output the plots from the papers, the statistical analysis, and the user behavior analysis.

## Usage Dash App for visualization

To test the visualization app locally, just run:

```
python3 app.py
```

Or simply compose the docker image:

```
docker-compose up
```
