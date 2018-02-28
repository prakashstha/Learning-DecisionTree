# Learning: DecisionTree
This was the second assignment given to us during our Artificial Intelligence course work. The goal of this assignment was to give us some experience implementing a supervised classifier, specifically, a decision tree. 

Our task was to implement the constructor and `classify` method in `DecisionTree.java`. 

In the constructor, a decision tree is created as follows:

1.  If all examples have the same label, a leaf node is created.
2.  If no features are remaining, a leaf node is created.
3.  Otherwise, the feature F with the highest information gain is identified. A branch node is created where for each possible value V of feature F:
    1.  The subset of examples where F=V is selected.
    2.  A decision (sub)tree is recursively created for the selected examples. None of these subtrees nor their descendants are allowed to branch again on feature F.

In `classify`, a prediction for a new example E is made as follows:

1.  For a leaf node where all examples have the same label, that label is returned.
2.  For a leaf node where the examples have more than one label, the most frequent label is returned.
3.  For a branch node based on a feature F, E is inspected to determine the value V that it has for feature F.
    1.  If the branch node has a subtree for V, then example E is recursively classified using the subtree.
    2.  If the branch node does not have a subtree for V, then the most frequent label for the examples at the branch node is returned.


