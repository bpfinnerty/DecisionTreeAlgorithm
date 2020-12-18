# DecisionTreeAlgorithm
@Author: Brian Finnerty

This project was used to test my knowledge of Decision tree algorithms and my implementation of AdaBoost. 

AdaBoost utilizes key splits in differences of the Dutch and English language build out a decision tree to decide if a phrase or sentence is English or Dutch.
By analyzing the entropy, the different identifiers, the algorithm chooses which item to split on and creates a node in the tree for it. This continues until 
the program has found a stump where it believes that the remaining data is either all English, all Dutch or goes with the known majority from the test data.

This algorithm takes training data in the form of sentences trailed by en or nl to denote either English or Dutch respectively as well as a location for the hypothesis file.
The second step is to input a new file where it will categorize each line as either Dutch or English.
