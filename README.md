_Details can be found in Report.pdf_

# Language Identification System

This is a language identification system which uses Discriminating between Similar Languages (DSL) Shared Task 2015 corpus. The corpus consists of a total of 26.000 sentences from 13 languages, each with 2000 sentences. The languages in question are Bulgarian, Bosnian, Czech, Argentine Spanish, Peninsular Spanish, Croatian, Indonesian, Macedonian, Malay, Brazilian Portuguese, European Portuguese, Slovak and Serbian.

The approach used in the identification is based on character unigrams, i.e character frequencies. The assumption is that characters and frequency of them differ from one language to another, so that we may identify languages by considering characters in the sentences. Punctutation marks are not considered. 

Two models are tried on this corpus to solve this problem: Generative Modeling (Naïve Bayes) and Discriminative Modeling (SVM). We implemented the Bayesian approach ourselves and used Cornell’s SVM multiclass library. 

The performance of the models are measured by accuracy, recall, precision, F-measure. 
The program is implemented in Python.

## Program Execution

Details can be found in the Report.pdf
