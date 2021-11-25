# Visualization MTG-MER dataset

Source code for visualisation platform presented to IEEE Transactions on Affective Computing 2021.

-- Juan Sebastián Gómez-Cañón, Nicolás Gutiérrez-Páez, Lorenzo Porcaro, Alastair Porter, Estefanía Cano, Perfecto Herrera, Aggelos Gkiokas, Patricia Santos, Davinia Hernández-Leo, Cynthia Liem, and Emilia Gómez

## Abstract

We present a data set for the analysis of personalized Music Emotion Recognition (MER) systems.
We developed the Music Enthusiasts platform aiming to improve the gathering and analysis of the so-called “ground truth” needed as input to such systems.
Our platform involves citizen science strategies to engage with audiences and generate large-scale music emotion annotations -- the platform presents didactic information and musical recommendations as incentivization, while collecting data regarding demographics, language, and mood from each participant.
Participants annotated each music excerpt with single free-text emotion word (in native language), forced-choice of distinct emotion categories, preference, and familiarity.
Additionally, participants stated the reasons for each annotation -- including those particular to emotion perception and emotion induction.
An extensive analysis of the participants' ratings is presented and a novel personalization strategy is evaluated.
Results evidence that using the “collective judgement” as prior knowledge for active learning allows for more effective personalization of MER systems for this particular data set.
Our data set is publicly available and we invite researchers to use it for testing MER systems.

## Usage

Simply compose the docker image:

```
docker-compose up
```

## Online Website

With UPF-VPN: http://mirlab-web1.s.upf.edu:8050/vis-mtg-mer

External: https://trompa-mtg.upf.edu:8050/vis-mtg-mer
