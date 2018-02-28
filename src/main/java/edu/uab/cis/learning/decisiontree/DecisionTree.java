package edu.uab.cis.learning.decisiontree;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;

import org.junit.Assert;



/**
 * A decision tree classifier.
 * 
 * @param <LABEL>
 *          The type of label that the classifier predicts.
 * @param <FEATURE_NAME>
 *          The type used for feature names.
 * @param <FEATURE_VALUE>
 *          The type used for feature values.
 */
public class DecisionTree<LABEL, FEATURE_NAME, FEATURE_VALUE> {

	//Information Gain of each feature
	Set<FEATURE_NAME> featuresSet;
	List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> orig_trainingDataList;
	Node root;
	  /**
   * Trains a decision tree classifier on the given training examples.
   * 
   * <ol>
   * <li>If all examples have the same label, a leaf node is created.</li>
   * <li>If no features are remaining, a leaf node is created.</li>
   * <li>Otherwise, the feature F with the highest information gain is
   * identified. A branch node is created where for each possible value V of
   * feature F:
   * <ol>
   * <li>The subset of examples where F=V is selected.</li>
   * <li>A decision (sub)tree is recursively created for the selected examples.
   * None of these subtrees nor their descendants are allowed to branch again on
   * feature F.</li>
   * </ol>
   * </li>
   * </ol>
   * 
   * @param trainingData
   *          The training examples, where each example is a set of features and
   *          the label that should be predicted for those features.
   */
  public DecisionTree(Collection<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> trainingData) {
	  List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> trainingDataList =new ArrayList<LabeledFeatures<LABEL,FEATURE_NAME,FEATURE_VALUE>>(trainingData);
	  orig_trainingDataList = new ArrayList<LabeledFeatures<LABEL,FEATURE_NAME,FEATURE_VALUE>>(trainingData);
	  featuresSet = new HashSet<FEATURE_NAME>();	// to hold all the feature name available in training data

	  root = constructTree(trainingDataList, new Node());
  }
  
  
  private Node constructTree(List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> trainingDataList, Node parentNode)
  {
	  //System.out.println("training data size" + trainingDataList.size());
	  Node nd = new Node();
	  nd.setParentNode(parentNode);
	  nd.setUsedFeatSet(parentNode.getUsedFeatureSet());
	  
	  nd.setTrainingData(trainingDataList);
	  
	  //check if all examples have same LABEL
	  Map<LABEL, Integer> lbl_count = new HashMap<LABEL, Integer>();
	  
	  // compute the frequency of each of the labels in training data
	  for(LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE> lbl_feature:trainingDataList)
	  {
		 
		  LABEL lbl = (LABEL)lbl_feature.getLabel();
		  if(lbl_count.containsKey(lbl))
		  {
			  int val = lbl_count.get(lbl);
			  lbl_count.put(lbl, val+1);
		  }
		  else{
			  lbl_count.put(lbl, 1);
		  }
//		  finding list of features
		  featuresSet.addAll(new ArrayList<FEATURE_NAME>(lbl_feature.getFeatureNames()));
	  }
	  
	
	  //System.out.println("Label size:" + lbl_count.size());
	  //System.out.println(lbl_count.entrySet());
	  
	  //if all examples have same label create leaf node
	  if(lbl_count.size()==1)
	  {
		  //all examples have same label -- create leaf node
		  nd.setLeaf(true);
		  nd.setChildren(new HashMap<FEATURE_VALUE,Node>());
		  nd.setFeature_name(null);
		  for(Map.Entry<LABEL, Integer> entry: lbl_count.entrySet())
		  {
			  nd.setLabel(entry.getKey());
			  //System.out.println("Used Label: " + nd.getLabel());
		  }
		  
		  //return nd;
		  
	  }
	  //still there are different LABEL 
	  else
	  {
		
		//find high probable label
		  LABEL probable_label = null;
		  int max = 0;
		  //find the feature with highest probability i.e. count
		  for(LABEL lbl: lbl_count.keySet())
		  {
			  int val = lbl_count.get(lbl);
			  if(max<val){
				  max = val;
				  probable_label = lbl;
			  }
		  }
		  nd.setLabel(probable_label);  
		  //System.out.println("Label used:" + nd.getLabel());
		  
		//option 1: No feature remaining... create leaf node with label  having highest count of label
		  if(nd.getUsedFeatureSet().size() == featuresSet.size())
		  {
			  nd.setLeaf(true);
			  nd.setFeature_name(null);
			  nd.setChildren(new HashMap<FEATURE_VALUE, Node>());
			  //nd.setLabel(probable_label);
		  }
		  //option2: feature remaining -- create subtree based on information gain
		  else
		  {
			  nd.setLeaf(false);
			  //nd.setLabel(probable_label);
			  Map<FEATURE_NAME, Double>IG = informationGain(trainingDataList, nd.getUsedFeatureSet());
			  Map<FEATURE_VALUE, Node> children = new HashMap<FEATURE_VALUE, Node>();
			  FEATURE_NAME max_feature_name = null;
			  Double maxIG = Double.NEGATIVE_INFINITY;
			  for(FEATURE_NAME feat_name: IG.keySet())
			  {
				  if(maxIG<IG.get(feat_name))
				  {
					  maxIG = IG.get(feat_name);
					  max_feature_name = feat_name;
				  }  
			  }
			  Set<FEATURE_NAME> fname_set = nd.getUsedFeatureSet();
			  if(max_feature_name!=null)
				  fname_set.add(max_feature_name);
			  nd.setUsedFeatSet(fname_set);
			  
			  nd.setFeature_name(max_feature_name);
			  //System.out.println("feature selected: "+max_feature_name);
			  Set<FEATURE_VALUE> uniqueFeatureVal = getUniqueFeatValueFromFeatureName(trainingDataList, max_feature_name);
			  //System.out.println("F_name: " + max_feature_name +"unique value: " + uniqueFeatureVal);
			  for(FEATURE_VALUE f_val: uniqueFeatureVal)
			  {
				  List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> filteredData = 
						  filterTrainingData(trainingDataList, max_feature_name, f_val);
				//  System.out.println("feature name ;" + max_feature_name + "\n feature value: " + f_val);
				  
				  //System.out.println("size of filtered data:" + filteredData.size());
				  Node child = constructTree(filteredData, nd);
				  children.put(f_val, child);
				  
				  
			  }
			  if(children!=null)
			  {
				  nd.setChildren(children);
			  }
		  }
		  
	  }
//	  if(nd.isLeaf())
//		  System.out.println("Leaf Node:" + nd.getLabel());
//	  else
//		  System.out.println("Feature Name: " + nd.getFeature_name());
	 
	//  System.out.println("lable used: "+nd.getLabel());
	  return nd;
  }
  
//  //from the label and its count find highest frequent label
//  private LABEL highFrequentLabel(Map<LABEL, Integer> lbl_count)
//  {
//	  int max = 0;
//	  LABEL probable_label = null;
//	  for(LABEL lbl: lbl_count.keySet())
//	  {
//		  int val = lbl_count.get(lbl);
//		  if(max<val){
//			  max = val;
//			  probable_label = lbl;
//		  }
//	  }
//	  return probable_label;
//  }
  
//Find the univque feature value of given feature name
  private Set<FEATURE_VALUE> getUniqueFeatValueFromFeatureName(
		List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> trainingDataList,
		FEATURE_NAME feature_name) {
	  Set<FEATURE_VALUE> uniqueVal = new HashSet<FEATURE_VALUE>();
	  for(LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE> lblFeature: trainingDataList)
	  {
		  FEATURE_VALUE val = lblFeature.getFeatureValue(feature_name);
		  
			  uniqueVal.add(val);
//		  Set<FEATURE_NAME> f_nameSet = lblFeature.getFeatureNames();
//		  for(FEATURE_NAME f_name: f_nameSet)
//		  {
//			  FEATURE_VALUE f_val = lblFeature.getFeatureValue(f_name);
//			  if(f_val != null)
//				  uniqueVal.add(f_val);
//		  }
	  }
	return uniqueVal;
}


/**
   * Predicts a label given a set of features.
   * 
   * <ol>
   * <li>For a leaf node where all examples have the same label, that label is
   * returned.</li>
   * <li>For a leaf node where the examples have more than one label, the most
   * frequent label is returned.</li>
   * <li>For a branch node based on a feature F, E is inspected to determine the
   * value V that it has for feature F.
   * <ol>
   * <li>If the branch node has a subtree for V, then example E is recursively
   * classified using the subtree.</li>
   * <li>If the branch node does not have a subtree for V, then the most
   * frequent label for the examples at the branch node is returned.</li>
   * </ol>
   * <li>
   * </ol>
   * 
   * @param features
   *          The features for which a label is to be predicted.
   * @return The predicted label.
   */
  public LABEL classify(Features<FEATURE_NAME, FEATURE_VALUE> features) {
	 return traverseTree(features, root);
  }
  int level = 0;
  private Map<FEATURE_NAME, Double> informationGain(List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> trainingData, Set<FEATURE_NAME>usedFeatureSet){
	 
	 // System.out.println("Level : " + (level++));
	  //System.out.println("Used Feature set"+usedFeatureSet.size());
//	  for(FEATURE_NAME f_name: usedFeatureSet)
//	  {
//		  System.out.println("Used:" + f_name);
//	  }
	  
	 //System.out.println("Size of data: " + trainingData.size());
	  
	  Map<FEATURE_NAME, Double> infoGain= new HashMap<FEATURE_NAME, Double>(); //tree map store the record by sorting
	  Set<FEATURE_NAME> featureNames = new HashSet<FEATURE_NAME>();
	  
	  // Finding remaining features' name
	  for(LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE> lblFeature: trainingData)
	  {
		  Set<FEATURE_NAME> tempFeatureName = lblFeature.getFeatureNames();
		  for(FEATURE_NAME f_name: tempFeatureName)
		  {
			  if(!usedFeatureSet.contains(f_name))
				  	featureNames.add(f_name);
		  }
		  
	  }
	  
	  /*----Finding unique values to each of the feature in used--------*/
	  
	  Map<FEATURE_NAME, Set<FEATURE_VALUE>> featureValueMap = new HashMap<FEATURE_NAME, Set<FEATURE_VALUE>>();
	  for(FEATURE_NAME featName: featureNames)
	  {
		  Set<FEATURE_VALUE> tempfeatureValue = new HashSet<FEATURE_VALUE>();
		  for(LabeledFeatures<LABEL,FEATURE_NAME, FEATURE_VALUE> lblFeatures: trainingData)
		  {
			  tempfeatureValue.add(lblFeatures.getFeatureValue(featName));
		  }
		  featureValueMap.put(featName, tempfeatureValue);
	  }
	  /*-------------- End of this section --------------*/
	
	  //computing h(y)
	  double hOfY = computeEntroyOfLabel(trainingData);
	  
	  
	  /*Computing entropy of for each of features i.e H(y|given)*/
	  //System.out.println("featureNames:"+featureNames);
	  for(FEATURE_NAME feat_name: featureNames){
		  Set<FEATURE_VALUE> featureValues = featureValueMap.get(feat_name);
		  double tempEntroy = 0.0;
		  for(FEATURE_VALUE feat_val: featureValues)
		  {
			  //finding probability of being feat_val P(X=x)
			  double pOfX = findPOfX(feat_name, feat_val, trainingData);
			  //System.out.println("P("+feat_name + ","+feat_val+")="+ pOfX);
			  //filter out the unwanted examples i.e filter examples containing only given feature value of given feature name
			  List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> filteredData = filterTrainingData(trainingData, feat_name, feat_val);
			  
			  
			  
			  double hOfyGiven = computeEntroyOfLabel(filteredData);
			  //System.out.println("H("+feat_name + ","+feat_val+")="+ hOfyGiven);
			  
			  tempEntroy += pOfX * hOfyGiven; 
		  }
		 //System.out.println(feat_name + ":" + tempEntroy);
		  infoGain.put(feat_name, hOfY-tempEntroy);	  
	  }
	 return infoGain;
  }
  
  
  private double findPOfX(FEATURE_NAME feat_name, FEATURE_VALUE feat_val,
		 List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> trainingData) {
	  // compute the frequency of each of the labels in training data
	  int feat_count = 0;
	  int total_count = 0;
	  for(LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE> lbl_feature:trainingData)
	  {
		 
		  FEATURE_VALUE f_val = lbl_feature.getFeatureValue(feat_name);
		  total_count++;
		  if(f_val.equals(feat_val))
			  feat_count++;
		  
	  }
	  
	  return feat_count/(double)total_count;
}


/*---------------Computes H(Y)--------------*/
  private double computeEntroyOfLabel(List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> examples)
  {
	  
	  Map<LABEL, Double> uniqueLabelCount = new HashMap<LABEL, Double>();
	  int dataSize = 0;
	  for (int i = 0; i < examples.size(); i++) {
		LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE> lblFeature = examples.get(i);
		LABEL tempLabel = (LABEL)lblFeature.getLabel();
		  if(uniqueLabelCount.containsKey(tempLabel))
		  {
			 Double val = uniqueLabelCount.remove(tempLabel);
			 uniqueLabelCount.put(tempLabel, val+1.0);
		  }
		  else{
			  uniqueLabelCount.put(tempLabel, 1.0);
		  }
		  dataSize++;
	}
	  
	  //computing h(y)
	  double hOfY = 0;
	  for(Map.Entry<LABEL, Double> entry: uniqueLabelCount.entrySet())
	  {
		  Double probPerLabel = entry.getValue()/dataSize;
		  hOfY -= probPerLabel*(Math.log(probPerLabel)/Math.log(2)); // takin log base 2
	  }
	  return hOfY;
  }
  //filter training data on the basis of given feature_value for given feature_name
  private List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> filterTrainingData(
		  List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>>  trainingData,
		  FEATURE_NAME featureName, FEATURE_VALUE featureValue)
  {
	  List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> filteredData = new ArrayList<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>>();
	  for(LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE> lblFeature: trainingData){
		  FEATURE_VALUE feat_Value = lblFeature.getFeatureValue(featureName);
		  
		  //*********CHECK IF WE NEED TO USE equals() METHOD OR == 
		  if(feat_Value.equals(featureValue))
		  {
			  filteredData.add(lblFeature);
		  }
	  }
	  //System.out.println("Unfiltered Data: " + trainingData.toArray());
	  //System.out.println("Feature Name : " + featureName + "\n Feature Value: " + featureValue);
	  //System.out.println(filteredData.toArray());
	  
	  return filteredData;
  }
  //filter training data on the basis of given feature_name
  private List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> filterTrainingData(List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>>  trainingData,
		  FEATURE_NAME featureName)
  {
	  List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> filteredData = new ArrayList<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>>();
	  for(LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE> lblFeature: trainingData){
		  FEATURE_VALUE feat_Value = lblFeature.getFeatureValue(featureName);
		  //*********CHECK IF WE NEED TO USE equals() METHOD OR == 
		  if(feat_Value!=null)
		  {
			  filteredData.add(lblFeature);
		  }
	  }
	  
	  return filteredData;
  }
  public LABEL traverseTree(Features<FEATURE_NAME, FEATURE_VALUE> features, Node node) {
	  	//if current node is leaf node return label of current node
	  	if(node.isLeaf()){
			return node.getLabel();
	  	}
	  	else{
	  		//System.out.println("Feature: " + node.getFeature_name());
			FEATURE_NAME f_name = node.getFeature_name();
			FEATURE_VALUE f_val = features.getFeatureValue(f_name);
			Map<FEATURE_VALUE, Node> children = node.getChildren();
			Node child = children.get(f_val);
			if(child == null)
				return node.getLabel();
			return traverseTree(features, child);
		}
		
	}
  private class Node{
	  private FEATURE_NAME feature_name;
	  private boolean leaf = false;
	  private Node parentNode;
	  private LABEL label;
	  private Map<FEATURE_VALUE,Node> children=null;
	  private Set<FEATURE_NAME> usedFeatureSet;
	  private List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> trainingData;
	  public Node()
	  {
		  this.trainingData = new ArrayList<LabeledFeatures<LABEL,FEATURE_NAME,FEATURE_VALUE>>();
		  this.usedFeatureSet = new HashSet<FEATURE_NAME>();
		  this.children = new HashMap<FEATURE_VALUE, Node>();
		  this.trainingData = new ArrayList<LabeledFeatures<LABEL,FEATURE_NAME,FEATURE_VALUE>>();
	  }
	
	  public void setParentNode(Node node){
		  this.parentNode = node;
	  }
	public FEATURE_NAME getFeature_name() {
		return feature_name;
	}
	
	public void setFeature_name(FEATURE_NAME feature_name) {
		this.feature_name = feature_name;
	}
	public Map<FEATURE_VALUE, Node> getChildren() {
		return children;
	}
	public void setChildren(Map<FEATURE_VALUE,Node> children) {
		this.children.putAll(children);
	}
	public boolean isLeaf() {
		return leaf;
	}
	public void setLeaf(boolean isLeaf) {
		this.leaf = isLeaf;
	}
	public LABEL getLabel() {
		return label;
	}
	public void setLabel(LABEL label) {
		this.label = label;
	}
	
	public void setTrainingData(
			List<LabeledFeatures<LABEL, FEATURE_NAME, FEATURE_VALUE>> trainingData) {
		this.trainingData.addAll(trainingData);
	} 
	public Set<FEATURE_NAME> getUsedFeatureSet()
	{
		return this.usedFeatureSet;
	}
	
	public void setUsedFeatSet(Set<FEATURE_NAME> usedFeatureSet)
	{
		this.usedFeatureSet.addAll(usedFeatureSet);
	}
	
	
  }
  public static void main(String[] args)
  {
	  List<LabeledFeatures<String, Integer, String>> trainingData = getTrainingDataFromAIBook();
	    DecisionTree<String, Integer, String> classifier = new DecisionTree<>(trainingData);

	    // frequent features
	    Features<Integer, String> testDatum = Features.of();
	    //Assert.assertEquals("yes", classifier.classify(testDatum));
	    // As at level hungry, majority data is no
	    testDatum = Features.of("yes", "no"  ,"no"  ,"", "full" );
	    //Assert.assertEquals("no", classifier.classify(testDatum));
	    // As at level Type, majority data is no or yes; a tie
	    testDatum = Features.of("no", "no"  ,"no"  ,"yes", "full", "", "", "", "SOME TYPE");
	    //Assert.assertEquals("yes", classifier.classify(testDatum));
	    
	    // test all paths
//	    Assert.assertEquals("no", classifier.classify(Features.of("no", "no", "no", "yes", "none", "1", "no", "no",  "french", "30-60")));
System.out.println(classifier.classify(Features.of("no", "no", "no", "yes", "none", "1", "no", "no",  "french", "30-60")));
	    //	    Assert.assertEquals("yes", classifier.classify(Features.of("yes", "no", "no", "yes", "some", "1", "no", "no", "french", "30-60")));
//	    Assert.assertEquals("no", classifier.classify(Features.of("yes", "no","no", "no" , "full", "1", "no", "no", "french", "30-60")));
//	    //Assert.assertEquals("yes", classifier.classify(Features.of("yes", "no","no", "yes" , "full", "1", "no", "no", "french", "30-60")));
//	    Assert.assertEquals("no", classifier.classify(Features.of("yes", "no","no", "yes" , "full", "1", "no", "no", "italian", "30-60")));
//	    Assert.assertEquals("yes", classifier.classify(Features.of("yes", "no","no", "yes" , "full", "1", "no", "no", "burger", "30-60")));
//	    Assert.assertEquals("no", classifier.classify(Features.of("yes", "no","no", "yes" , "full", "1", "no", "no", "thai", "30-60")));
//	    Assert.assertEquals("yes", classifier.classify(Features.of("yes", "no","yes", "yes" , "full", "1", "no", "no", "thai", "30-60")));

  }

  //  @formatter:off
  private static List<LabeledFeatures<String, Integer, String>> getTrainingDataFromAIBook() {
    List<LabeledFeatures<String, Integer, String>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofStrings("yes","yes","no"  ,"no"  ,"yes" ,"some" ,"3" ,"no"  ,"yes","french" ,"0-10"));
    trainingData.add(LabeledFeatures.ofStrings("no" ,"yes","no"  ,"no"  ,"yes" ,"full" ,"1" ,"no"  ,"no" ,"thai"   ,"30-60"));
    trainingData.add(LabeledFeatures.ofStrings("yes","no" ,"yes" ,"no"  ,"no"  ,"some" ,"1" ,"no"  ,"no" ,"burger" ,"0-10"));
    trainingData.add(LabeledFeatures.ofStrings("yes","yes","no"  ,"yes" ,"yes" ,"full" ,"1" ,"yes" ,"no" ,"thai"   ,"10-30"));
    trainingData.add(LabeledFeatures.ofStrings("no" ,"yes","no"  ,"yes" ,"no" ,"full"  ,"3" ,"no"  ,"yes","french" ,">60"));
    trainingData.add(LabeledFeatures.ofStrings("yes","no" ,"yes" ,"no"  ,"yes" ,"some" ,"2" ,"yes" ,"yes","italian","0-10"));
    trainingData.add(LabeledFeatures.ofStrings("no" ,"no" ,"yes" ,"no"  ,"no" ,"none"  ,"1" ,"yes" ,"no" ,"burger" ,"0-10"));
    trainingData.add(LabeledFeatures.ofStrings("yes","no" ,"no"  ,"no"  ,"yes" ,"some" ,"2" ,"yes" ,"yes","thai"   ,"0-10"));
    trainingData.add(LabeledFeatures.ofStrings("no" ,"no" ,"yes" ,"yes" ,"no" ,"full"  ,"1" ,"yes" ,"no" ,"burger" ,">60"));
    trainingData.add(LabeledFeatures.ofStrings("no" ,"yes","yes" ,"yes" ,"yes" ,"full" ,"3" ,"no"  ,"yes","italian","10-30"));
    trainingData.add(LabeledFeatures.ofStrings("no" ,"no" ,"no"  ,"no"  ,"no" ,"none"  ,"1" ,"no"  ,"no" ,"thai"   ,"0-10"));
    trainingData.add(LabeledFeatures.ofStrings("yes","yes","yes" ,"yes" ,"yes" ,"full" ,"1" ,"no"  ,"no" ,"burger" ,"30-60"));    
    return trainingData;
  }
  
}
