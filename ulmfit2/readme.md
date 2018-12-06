# ULMFiT 2 Arabic 

This folder contains scripts to train Ar corpus and language model based on most recent multlingual ulmfit 
from https://github.com/n-waves/ulmfit-multilingual/tree/refactor  
Some code sections has been addjusted to fit narrow cases or hard-wire parameters. A notebook may be posted in this folder to provide a roadmap.  
All tests were done on Google colab (free instance). Model training currently works with QRNN and bidir off (qrnn works at model creation level but fastai doesn't seem to read it well). Also, unfreeze is probably acting strangely.

