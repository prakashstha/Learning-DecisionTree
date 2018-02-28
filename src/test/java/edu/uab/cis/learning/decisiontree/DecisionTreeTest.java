package edu.uab.cis.learning.decisiontree;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;

import org.junit.Assert;
import org.junit.Test;

public class DecisionTreeTest {

  @Test
  public void testMostFrequentClass() {
    // assemble training data
    List<LabeledFeatures<Integer, Integer, String>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofStrings(0));
    trainingData.add(LabeledFeatures.ofStrings(1));
    trainingData.add(LabeledFeatures.ofStrings(1));
    // train the classifier
    DecisionTree<Integer, Integer, String> classifier = new DecisionTree<>(trainingData);
    // test that the classifier returns the most frequent class
    Features<Integer, String> testDatum = Features.of();
    Assert.assertEquals(Integer.valueOf(1), classifier.classify(testDatum));
  }

  @Test
  public void testBigData() throws FileNotFoundException, IOException {
    // assemble training data
    List<LabeledFeatures<String, Integer, String>> trainingData = new ArrayList<>();
    File f = new File("/AI Assignment/DecisionTree/learning-decisiontree/src/test/java/edu/uab/cis/learning/decisiontree/data.csv");
    FileInputStream fis = new FileInputStream(f);
    BufferedReader reader = new BufferedReader(new InputStreamReader(fis));
    String line = new String();
    while ((line = reader.readLine()) != null) {
      StringTokenizer st = new StringTokenizer(line, ",");
      String[] arr = new String[10];
      for (int i = 0; i < 10; i++) {
        arr[i] = (st.nextToken());
      }
      trainingData.add(LabeledFeatures.ofStrings(
          arr[9],
          arr[0],
          arr[1],
          arr[2],
          arr[3],
          arr[4],
          arr[5],
          arr[6],
          arr[7],
          arr[8]));
    }
    // train the classifier
    DecisionTree<String, Integer, String> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals(
        "0",
        classifier.classify(Features.of("2", "1", "3", "2", "2", "1", "2", "1", "1")));
  }

  @Test
  public void testFullyPredictiveFeature() {
    // assemble training data
    List<LabeledFeatures<String, Integer, Integer>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0));
    trainingData.add(LabeledFeatures.ofIntegers("A", 1, 0));
    trainingData.add(LabeledFeatures.ofIntegers("B", 0, 1));
    // train the classifier
    DecisionTree<String, Integer, Integer> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("B", classifier.classify(Features.of(1, 1)));
  }

  @Test
  public void testFullyPredictiveFeaturefromlecture() {
    // assemble training data
    List<LabeledFeatures<String, Integer, String>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "a"));
    trainingData.add(LabeledFeatures.ofStrings("false", "0", "b"));
    trainingData.add(LabeledFeatures.ofStrings("true", "1", "c"));
    trainingData.add(LabeledFeatures.ofStrings("false", "0", "c"));
    trainingData.add(LabeledFeatures.ofStrings("true", "1", "a"));
    // train the classifier
    DecisionTree<String, Integer, String> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("false", classifier.classify(Features.of("1", "b")));
  }

  @Test
  public void testFullyPredictiveFeaturefromlecture1() {
    // assemble training data
    List<LabeledFeatures<String, Integer, Integer>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 1, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 2, 1));
    trainingData.add(LabeledFeatures.ofIntegers("C", 1, 1, 2));
    trainingData.add(LabeledFeatures.ofIntegers("C", 2, 2, 1));
    // train the classifier
    DecisionTree<String, Integer, Integer> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("C", classifier.classify(Features.of(1, 1, 2)));
  }

  @Test
  public void testFullyPredictiveFeaturefromAmitVai() {
    // assemble training data
    List<LabeledFeatures<String, Integer, String>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "a"));
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "b"));
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "c"));
    trainingData.add(LabeledFeatures.ofStrings("true", "1", "a"));
    trainingData.add(LabeledFeatures.ofStrings("false", "1", "b"));
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "e"));
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "f"));
    trainingData.add(LabeledFeatures.ofStrings("false", "1", "c"));
    // train the classifier
    DecisionTree<String, Integer, String> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("true", classifier.classify(Features.of(" ", " ")));
  }

  @Test
  public void testFullyPredictiveFeaturefromAmitVai1() {
    // assemble training data
    List<LabeledFeatures<String, Integer, String>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "a"));
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "b"));
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "c"));
    trainingData.add(LabeledFeatures.ofStrings("true", "1", "a"));
    trainingData.add(LabeledFeatures.ofStrings("false", "1", "b"));
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "e"));
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "f"));
    trainingData.add(LabeledFeatures.ofStrings("false", "1", "c"));
    // train the classifier
    DecisionTree<String, Integer, String> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("true", classifier.classify(Features.of("", "")));
  }

  @Test
  public void testFullyPredictiveFeaturefromAmitVai2() {
    // assemble training data
    List<LabeledFeatures<String, Integer, String>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "a"));
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "b"));
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "c"));
    trainingData.add(LabeledFeatures.ofStrings("false", "1", "a"));
    trainingData.add(LabeledFeatures.ofStrings("false", "1", "b"));
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "e"));
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "f"));
    trainingData.add(LabeledFeatures.ofStrings("true", "1", "e"));
    // train the classifier
    DecisionTree<String, Integer, String> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("false", classifier.classify(Features.of("1", "c")));
  }

  @Test
  public void testFullyPredictiveFeaturefromAmitVai3() {
    // assemble training data
    List<LabeledFeatures<String, Integer, String>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "a"));
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "b"));
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "c"));
    trainingData.add(LabeledFeatures.ofStrings("true", "1", "a"));
    trainingData.add(LabeledFeatures.ofStrings("false", "1", "b"));
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "e"));
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "f"));
    trainingData.add(LabeledFeatures.ofStrings("false", "1", "c"));
    // train the classifier
    DecisionTree<String, Integer, String> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("false", classifier.classify(Features.of("1", "x")));
  }

  @Test
  public void testFullyPredictiveFeatureBig() {
    // assemble training data
    List<LabeledFeatures<String, Integer, String>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofStrings("n", "r", "h", "h", "f"));
    trainingData.add(LabeledFeatures.ofStrings("n", "r", "h", "h", "t"));
    trainingData.add(LabeledFeatures.ofStrings("y", "o", "h", "h", "f"));
    trainingData.add(LabeledFeatures.ofStrings("y", "s", "m", "h", "f"));
    trainingData.add(LabeledFeatures.ofStrings("y", "s", "c", "n", "f"));
    trainingData.add(LabeledFeatures.ofStrings("n", "s", "c", "n", "t"));
    trainingData.add(LabeledFeatures.ofStrings("y", "o", "c", "n", "t"));
    trainingData.add(LabeledFeatures.ofStrings("n", "r", "m", "h", "f"));
    trainingData.add(LabeledFeatures.ofStrings("y", "r", "c", "n", "f"));
    trainingData.add(LabeledFeatures.ofStrings("y", "s", "m", "n", "f"));
    trainingData.add(LabeledFeatures.ofStrings("y", "r", "m", "n", "t"));
    trainingData.add(LabeledFeatures.ofStrings("y", "o", "m", "h", "t"));
    trainingData.add(LabeledFeatures.ofStrings("y", "o", "h", "n", "f"));
    trainingData.add(LabeledFeatures.ofStrings("n", "s", "m", "h", "t"));
    // train the classifier
    DecisionTree<String, Integer, String> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("n", classifier.classify(Features.of("r", "m", "h", "t")));
  }

  @Test
  public void testWeatherFeature() {
    List<LabeledFeatures<String, Integer, String>> trainingData = getTemparatureTrainingData();
    DecisionTree<String, Integer, String> classifier = new DecisionTree<>(trainingData);
    // test all paths
    Assert.assertEquals("yes", classifier.classify(Features.of("overcast")));
    // Assert.assertEquals("yes", classifier.classify(Features.of("sunny",
    // "hot", "high", "false")));
    // Assert.assertEquals("no", classifier.classify(Features.of("sunny", "hot",
    // "high", "true")));
    // Assert.assertEquals("no", classifier.classify(Features.of("rainy", "hot",
    // "high", "false")));
    // Assert.assertEquals("yes", classifier.classify(Features.of("rainy",
    // "hot", "normal", "false")));
  }

  private static List<LabeledFeatures<String, Integer, String>> getTemparatureTrainingData() {
    List<LabeledFeatures<String, Integer, String>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofStrings("no", "rainy", "hot", "high", "false"));
    trainingData.add(LabeledFeatures.ofStrings("no", "rainy", "hot", "high", "true"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "overcast", "hot", "high", "false"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "sunny", "mild", "high", "false"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "sunny", "cool", "normal", "false"));
    trainingData.add(LabeledFeatures.ofStrings("no", "sunny", "cool", "normal", "true"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "overcast", "cool", "normal", "true"));
    trainingData.add(LabeledFeatures.ofStrings("no", "rainy", "mild", "high", "false"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "rainy", "cool", "normal", "false"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "sunny", "mild", "normal", "false"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "rainy", "mild", "normal", "true"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "overcast", "mild", "high", "true"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "overcast", "hot", "normal", "false"));
    trainingData.add(LabeledFeatures.ofStrings("no", "sunny", "mild", "high", "true"));
    return trainingData;
  }

  @Test
  public void testMostFrequentClass1() {
    // assemble training data
    List<LabeledFeatures<Integer, Integer, String>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofStrings(0));
    trainingData.add(LabeledFeatures.ofStrings(1));
    trainingData.add(LabeledFeatures.ofStrings(1));
    // train the classifier
    DecisionTree<Integer, Integer, String> classifier = new DecisionTree<>(trainingData);
    // test that the classifier returns the most frequent class
    Features<Integer, String> testDatum = Features.of();
    Assert.assertEquals(Integer.valueOf(1), classifier.classify(testDatum));
  }

  @Test
  public void testFullyPredictiveFeature1() {
    // assemble training data
    List<LabeledFeatures<String, Integer, Integer>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0));
    trainingData.add(LabeledFeatures.ofIntegers("A", 1, 0));
    trainingData.add(LabeledFeatures.ofIntegers("B", 0, 1));
    // train the classifier
    DecisionTree<String, Integer, Integer> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("B", classifier.classify(Features.of(1, 1)));
  }

  @Test
  public void testInClassExample0() { // assemble training data
    List<LabeledFeatures<String, Integer, Integer>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 1, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 2, 1));
    trainingData.add(LabeledFeatures.ofIntegers("C", 1, 1, 2));
    trainingData.add(LabeledFeatures.ofIntegers("C", 2, 2, 1)); // train the
    DecisionTree<String, Integer, Integer> classifier = new DecisionTree<>(trainingData); // test
                                                                                          // that
                                                                                          // the
                                                                                          // classifier
                                                                                          // split
                                                                                          // on
                                                                                          // the
    Assert.assertEquals("A", classifier.classify(Features.of(0, 1, 1)));
  }

  @Test
  public void testInClassExample1() { // assemble training data
    List<LabeledFeatures<String, Integer, Integer>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 1, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 2, 1));
    trainingData.add(LabeledFeatures.ofIntegers("C", 1, 1, 2));
    trainingData.add(LabeledFeatures.ofIntegers("C", 2, 2, 1)); // train the
    DecisionTree<String, Integer, Integer> classifier = new DecisionTree<>(trainingData); // test
                                                                                          // that
                                                                                          // the
                                                                                          // classifier
                                                                                          // split
                                                                                          // on
                                                                                          // the
    Assert.assertEquals("C", classifier.classify(Features.of(2, 0, 1)));
  }

  @Test
  public void testInClassExample2() {
    // assemble training data
    List<LabeledFeatures<String, Integer, Integer>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 1, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 2, 1));
    trainingData.add(LabeledFeatures.ofIntegers("C", 1, 1, 2));
    trainingData.add(LabeledFeatures.ofIntegers("C", 2, 2, 1));
    // train the classifier
    DecisionTree<String, Integer, Integer> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("C", classifier.classify(Features.of(1, 0, 2)));
  }

  @Test
  public void testInClassExample3() {
    // assemble training data
    List<LabeledFeatures<String, Integer, Integer>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 1, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 2, 1));
    trainingData.add(LabeledFeatures.ofIntegers("C", 1, 1, 2));
    trainingData.add(LabeledFeatures.ofIntegers("C", 2, 2, 1));
    // train the classifier
    DecisionTree<String, Integer, Integer> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("B", classifier.classify(Features.of(1, 0, 1)));
  }

  @Test
  public void testInClassExample4() {
    // assemble training data
    List<LabeledFeatures<String, Integer, Integer>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 1, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 2, 1));
    trainingData.add(LabeledFeatures.ofIntegers("C", 1, 1, 2));
    trainingData.add(LabeledFeatures.ofIntegers("C", 2, 2, 1));
    // train the classifier
    DecisionTree<String, Integer, Integer> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("B", classifier.classify(Features.of(1, 0, 0)));
  }

  @Test
  public void testSimpleNoRemainingExamples() {
    // assemble training data
    List<LabeledFeatures<String, Integer, Integer>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofIntegers("A", 0));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1));
    trainingData.add(LabeledFeatures.ofIntegers("C", 1));
    // train the classifier
    DecisionTree<String, Integer, Integer> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("A", classifier.classify(Features.of(0)));
  }

  @Test
  public void testSameLabelOnSeperateBranchesRemainingExamplesC() {
    // assemble training data
    List<LabeledFeatures<String, Integer, Integer>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofIntegers("A", 1, 1, 0));
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("C", 1, 1, 1));
    // train the classifier
    DecisionTree<String, Integer, Integer> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("C", classifier.classify(Features.of(0, 1, 1)));
  }

  @Test
  public void testSameLabelOnSeperateBranchesRemainingExamplesAx2() {
    // assemble training data
    List<LabeledFeatures<String, Integer, Integer>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofIntegers("A", 1, 1, 0));
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("C", 1, 1, 1));
    // train the classifier
    DecisionTree<String, Integer, Integer> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("A", classifier.classify(Features.of(0, 1, 0)));
  }

  @Test
  public void testSameLabelOnSeperateBranchesRemainingExamplesAx0() {
    // assemble training data
    List<LabeledFeatures<String, Integer, Integer>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofIntegers("A", 1, 1, 0));
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("C", 1, 1, 1));
    // train the classifier
    DecisionTree<String, Integer, Integer> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("A", classifier.classify(Features.of(0, 0, 2)));
  }

  @Test
  public void testSameLabelOnSeperateBranchesRemainingExamplesB() {
    // assemble training data
    List<LabeledFeatures<String, Integer, Integer>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofIntegers("A", 1, 1, 0));
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("C", 1, 1, 1));
    // train the classifier
    DecisionTree<String, Integer, Integer> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("B", classifier.classify(Features.of(1, 0, 2)));
  }

  @Test
  public void testUnseenLabelOnSeperateBranchesRemainingExamples() {
    // assemble training data
    List<LabeledFeatures<String, Integer, Integer>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofIntegers("A", 1, 1, 0));
    trainingData.add(LabeledFeatures.ofIntegers("A", 0, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("B", 1, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers("C", 1, 1, 1));
    // train the classifier
    DecisionTree<String, Integer, Integer> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    Assert.assertEquals("A", classifier.classify(Features.of(1, 2, 2)));
  }

  @Test
  public void testTime() {
    // assemble training data
    List<LabeledFeatures<Integer, Integer, Integer>> trainingData = new ArrayList<>();
    Random rand = new Random();

    for (int n = 0; n < 1000; n++) {
      Integer[] randomFeatures = new Integer[100];

      for (int i = 0; i < randomFeatures.length; i++)
        randomFeatures[i] = rand.nextInt(10);

      trainingData.add(LabeledFeatures.ofIntegers(rand.nextInt(5), randomFeatures));
    }

    // take the time
    long timeafter;
    long timebefore = System.nanoTime();

    // train the classifier
    DecisionTree<Integer, Integer, Integer> classifier = new DecisionTree<>(trainingData);

    // take the time afterwards
    timeafter = System.nanoTime();

    // test that the classifier did not take longer than a JVM second to build
    // the tree
    System.out.println((timeafter - timebefore));
    Assert.assertTrue((timeafter - timebefore) < (long) 1e9);
  }

  @Test
  public void testClassEx1() {
    // assemble training data
    List<LabeledFeatures<String, Integer, String>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofStrings("T", "0", "a"));
    trainingData.add(LabeledFeatures.ofStrings("F", "0", "b"));
    trainingData.add(LabeledFeatures.ofStrings("T", "1", "c"));
    trainingData.add(LabeledFeatures.ofStrings("F", "0", "c"));
    trainingData.add(LabeledFeatures.ofStrings("F", "1", "b"));
    trainingData.add(LabeledFeatures.ofStrings("T", "1", "a"));
    // train the classifier
    DecisionTree<String, Integer, String> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    // Assert.assertEquals("B", classifier.classify(Features.of(1, 1)));
  }

  @Test
  public void testClassExercise() {
    List<LabeledFeatures<Boolean, Integer, Integer>> trainingData = new ArrayList<>();
    // x1 = {0 , 1} , x2 = {0, 1, 2}
    trainingData.add(LabeledFeatures.ofIntegers(Boolean.TRUE, 0, 0));
    trainingData.add(LabeledFeatures.ofIntegers(Boolean.FALSE, 0, 1));
    trainingData.add(LabeledFeatures.ofIntegers(Boolean.TRUE, 1, 2));
    trainingData.add(LabeledFeatures.ofIntegers(Boolean.FALSE, 0, 2));
    trainingData.add(LabeledFeatures.ofIntegers(Boolean.FALSE, 1, 1));
    trainingData.add(LabeledFeatures.ofIntegers(Boolean.TRUE, 1, 0));

    DecisionTree<Boolean, Integer, Integer> decisionTree = new DecisionTree<>(trainingData);
    Assert.assertEquals(Boolean.TRUE, decisionTree.classify(Features.of(1, 0)));
    Assert.assertEquals(Boolean.TRUE, decisionTree.classify(Features.of(0, 0)));
    Assert.assertEquals(Boolean.TRUE, decisionTree.classify(Features.of(4, 0)));

    Assert.assertEquals(Boolean.FALSE, decisionTree.classify(Features.of(0, 1)));
    Assert.assertEquals(Boolean.FALSE, decisionTree.classify(Features.of(1, 1)));

    Assert.assertEquals(Boolean.FALSE, decisionTree.classify(Features.of(0, 2)));
    Assert.assertEquals(Boolean.TRUE, decisionTree.classify(Features.of(1, 2)));

  }

  @Test
  public void testOffBranchFeatureValue() {
    String[][] dataSet =
        { { "USA", "yes", "scifi", "Success" }, { "USA", "no", "comedy", "Failure" },
            { "USA", "yes", "comedy", "Success" }, { "Europe", "no", "comedy", "Success" },
            { "Europe", "yes", "scifi", "Failure" }, { "Europe", "yes", "romance", "Failure" },
            { "Australia", "yes", "comedy", "Failure" }, { "Brazil", "no", "scifi", "Failure" },
            { "Europe", "yes", "comedy", "Success" }, { "USA", "yes", "comedy", "Success" } };
    List<LabeledFeatures<String, Integer, String>> trainingData = new ArrayList<>();

    for (int i = 0; i < 10; i++) {
      trainingData.add(LabeledFeatures.ofStrings(
          dataSet[i][3],
          dataSet[i][0],
          dataSet[i][2],
          dataSet[i][1]));
    }

    DecisionTree<String, Integer, String> decisionTree = new DecisionTree<>(trainingData);
    Assert.assertEquals("Success", decisionTree.classify(Features.of("USA", "scifi")));
  }

  @Test
  public void testBigDataSet() {
    List<LabeledFeatures<String, Integer, Integer>> trainingData = new ArrayList<>();
    Integer[][] totalData = new Integer[1000][100];
    String[] result = new String[1000];
    String[] resultSet = { "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N" };
    for (int i = 0; i < 7; i++) {
      for (int j = 0; j < 100; j++) {
        totalData[i * 100 + j][0] = i;
        result[i * 100 + j] = resultSet[i];
      }
    }

    Random random = new Random(Calendar.getInstance().getTimeInMillis());
    for (int i = 700; i < 1000; i++) {
      totalData[i][0] = random.nextInt(10) + 7;
      result[i] = String.valueOf(resultSet[random.nextInt(7) + 7]);
    }

    for (int i = 0; i < 1000; i++) {
      random = new Random(Calendar.getInstance().getTimeInMillis());
      for (int j = 1; j < 100; j++) {
        totalData[i][j] = random.nextInt(40) + 1;
      }
      trainingData.add(LabeledFeatures.ofIntegers(result[i], totalData[i]));
    }

    DecisionTree<String, Integer, Integer> decisionTree = new DecisionTree<>(trainingData);

    for (int i = 0; i < 700; i += 100)
      Assert.assertEquals(result[i], decisionTree.classify(Features.of(totalData[i])));

  }
  //------------ Khan -------------
  @Test
  public void testFullyPredictiveFeature2() {
      // assemble training data
      List<LabeledFeatures<String, Integer, Integer>> trainingData = new ArrayList<>();
      trainingData.add(LabeledFeatures.ofIntegers("A", 1, 10, 100));
      trainingData.add(LabeledFeatures.ofIntegers("A", 1, 10, 100));
      trainingData.add(LabeledFeatures.ofIntegers("B", 1, 20, 100));
      trainingData.add(LabeledFeatures.ofIntegers("A", 2, 20, 200));
      trainingData.add(LabeledFeatures.ofIntegers("B", 2, 30, 300));
      trainingData.add(LabeledFeatures.ofIntegers("B", 3, 30, 300));
      trainingData.add(LabeledFeatures.ofIntegers("A", 4, 30, 400));
      trainingData.add(LabeledFeatures.ofIntegers("B", 4, 40, 400));
      // train the classifier
      DecisionTree<String, Integer, Integer> classifier = new DecisionTree<>(trainingData);
      // test that the classifier split on the second feature
      Assert.assertEquals("B", classifier.classify(Features.of(2, 30, 200)));
  }

  // kumar

 

  // @formatter:on
  @Test
  public void testKDDMBookFeature() {
    List<LabeledFeatures<String, Integer, String>> trainingData = getTraininingDataFromKDDMbook();
    DecisionTree<String, Integer, String> classifier = new DecisionTree<>(trainingData);

    // frequent features
    Features<Integer, String> testDatum = Features.of();
    Assert.assertEquals("yes", classifier.classify(testDatum));
    testDatum = Features.of("Some arbitrary value :)");
    Assert.assertEquals("yes", classifier.classify(testDatum));

    // As at level youth, majority data is no
    testDatum = Features.of("youth", "high");
    Assert.assertEquals("no", classifier.classify(testDatum));
    testDatum = Features.of("senior", "high", "yes");
    // As at level senior, majority data is yes
    testDatum = Features.of("senior", "high");
    Assert.assertEquals("yes", classifier.classify(testDatum));
    // As at level senior, majority data is yes
    testDatum = Features.of("senior", "high", "yes", "very low");
    Assert.assertEquals("yes", classifier.classify(testDatum));

    // fully predictive features
    testDatum = Features.of("youth", "high", "no", "fair");
    Assert.assertEquals("no", classifier.classify(testDatum));

    // test all paths in tree
    testDatum = Features.of("youth", "Any value", "no", "download");
    Assert.assertEquals("no", classifier.classify(testDatum));
    testDatum = Features.of("youth", "Any value", "yes", "download");
    Assert.assertEquals("yes", classifier.classify(testDatum));
    testDatum = Features.of("middle_aged", "Any value", "whatever", "download");
    Assert.assertEquals("yes", classifier.classify(testDatum));
    testDatum = Features.of("senior", "Any value", "no", "excellent");
    // Book Decision is wrong here
    Assert.assertEquals("no", classifier.classify(testDatum));
    testDatum = Features.of("senior", "Any value", "no", "fair");
    // Book Decision is wrong here
    Assert.assertEquals("yes", classifier.classify(testDatum));
  }

  // @formatter:off
  
  @Test
  public void testAIFeature(){
    List<LabeledFeatures<String, Integer, String>> trainingData = getTrainingDataFromAIBook();
    DecisionTree<String, Integer, String> classifier = new DecisionTree<>(trainingData);

    // frequent features
    Features<Integer, String> testDatum = Features.of();
    //Assert.assertEquals("yes", classifier.classify(testDatum));
    // As at level hungry, majority data is no
    testDatum = Features.of("yes", "no"  ,"no"  ,"", "full" );
    Assert.assertEquals("no", classifier.classify(testDatum));
    // As at level Type, majority data is no or yes; a tie
    testDatum = Features.of("no", "no"  ,"no"  ,"yes", "full", "", "", "", "SOME TYPE");
    //Assert.assertEquals("yes", classifier.classify(testDatum));
    
    // test all paths
    Assert.assertEquals("no", classifier.classify(Features.of("no", "no", "no", "yes", "none", "1", "no", "no",  "french", "30-60")));
    Assert.assertEquals("yes", classifier.classify(Features.of("yes", "no", "no", "yes", "some", "1", "no", "no", "french", "30-60")));
    Assert.assertEquals("no", classifier.classify(Features.of("yes", "no","no", "no" , "full", "1", "no", "no", "french", "30-60")));
    //Assert.assertEquals("yes", classifier.classify(Features.of("yes", "no","no", "yes" , "full", "1", "no", "no", "french", "30-60")));
    Assert.assertEquals("no", classifier.classify(Features.of("yes", "no","no", "yes" , "full", "1", "no", "no", "italian", "30-60")));
    Assert.assertEquals("yes", classifier.classify(Features.of("yes", "no","no", "yes" , "full", "1", "no", "no", "burger", "30-60")));
    Assert.assertEquals("no", classifier.classify(Features.of("yes", "no","no", "yes" , "full", "1", "no", "no", "thai", "30-60")));
    Assert.assertEquals("yes", classifier.classify(Features.of("yes", "no","yes", "yes" , "full", "1", "no", "no", "thai", "30-60")));        
  }
  

  
  // @formatter:on

  @Test
  public void testSlideFeature() {
    // assemble training data
    List<LabeledFeatures<String, Integer, String>> trainingData = getTrainingDataFromSlides1();
    DecisionTree<String, Integer, String> classifier = new DecisionTree<>(trainingData);
    // test that the classifier split on the second feature
    // Assert.assertEquals("B", classifier.classify(Features.of(1, 1)));
  }

 







  private static List<LabeledFeatures<String, Integer, String>> getTrainingDataFromSlides1() {
    List<LabeledFeatures<String, Integer, String>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofStrings("true", "0", "a"));
    trainingData.add(LabeledFeatures.ofStrings("false", "0", "b"));
    trainingData.add(LabeledFeatures.ofStrings("true", "1", "c"));
    trainingData.add(LabeledFeatures.ofStrings("false", "0", "c"));
    trainingData.add(LabeledFeatures.ofStrings("false", "1", "b"));
    trainingData.add(LabeledFeatures.ofStrings("true", "1", "a"));
    return trainingData;
  }

  private static List<LabeledFeatures<String, Integer, String>> getTraininingDataFromKDDMbook() {
    List<LabeledFeatures<String, Integer, String>> trainingData = new ArrayList<>();
    trainingData.add(LabeledFeatures.ofStrings("no", "youth", "high", "no", "fair"));
    trainingData.add(LabeledFeatures.ofStrings("no", "youth", "high", "no", "excellent"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "middle_aged", "high", "no", "fair"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "senior", "medium", "no", "fair"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "senior", "low", "yes", "fair"));
    trainingData.add(LabeledFeatures.ofStrings("no", "senior", "low", "yes", "excellent"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "middle_aged", "low", "yes", "excellent"));
    trainingData.add(LabeledFeatures.ofStrings("no", "youth", "medium", "no", "fair"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "youth", "low", "yes", "fair"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "senior", "medium", "yes", "fair"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "youth", "medium", "yes", "excellent"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "middle_aged", "medium", "no", "excellent"));
    trainingData.add(LabeledFeatures.ofStrings("yes", "middle_aged", "high", "yes", "fair"));
    trainingData.add(LabeledFeatures.ofStrings("no", "senior", "medium", "no", "excellent"));
    return trainingData;
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
  


   


   
    @Test()
    public void testFromSlide24() {
  //  > Index x1 x2 f(x)
  //  > 0      0  a  true
  //  > 1      0  b  false
  //  > 2      1  c  true
  //  > 3      0  c  false
  //  > 4      1  b  false
  //  > 5      1  a  true
   List<LabeledFeatures<Boolean, Integer, Object>> trainingData = new ArrayList<>();
   trainingData.add(LabeledFeatures.<Boolean,Object>of(true, 0, 'a'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(false, 0, 'b'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(true, 1, 'c'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(false, 0, 'c'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(false, 1, 'b'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(true, 1, 'a'));
   DecisionTree<Boolean, Integer, Object> classifier = new DecisionTree<>(trainingData);
   
   
   Assert.assertEquals(true, classifier.classify(Features.<Object>of(0, 'a')));
   Assert.assertEquals(false, classifier.classify(Features.<Object>of(0, 'b')));  
   Assert.assertEquals(true, classifier.classify(Features.<Object>of(1, 'c')));  
   Assert.assertEquals(false, classifier.classify(Features.<Object>of(0, 'c')));  
   Assert.assertEquals(false, classifier.classify(Features.<Object>of(1, 'b')));  
   Assert.assertEquals(true, classifier.classify(Features.<Object>of(1, 'a'))); 

    }

    @Test()
    public void myTest1() {
   List<LabeledFeatures<Integer, Integer, Object>> trainingData = new ArrayList<>();
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, 0, 'a'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, 0, 'b'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, 0, 'c'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, 0, 'd'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, 0, 'e'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, 0, 'f'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, 0, 'g'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(0, 1, 'e'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(0, 1, 'f'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(0, 1, 'g'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(0, 1, 'h'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(0, 1, 'i'));
   //1st feature selected because '0' always leads to '1'
   DecisionTree<Integer, Integer, Object> classifier = new DecisionTree<>(trainingData);
   Assert.assertEquals(Integer.valueOf(1), classifier.classify(Features.<Object>of(0, 'i')));
    }
      
    @Test()  
    public void myTest2() {
   List<LabeledFeatures<Integer, Integer, Object>> trainingData = new ArrayList<>();
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, 0, 'a'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, 1, 'a'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, 2, 'a'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, 3, 'a'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, 4, 'a'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, 5, 'a'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, 6, 'a'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(0, 7, 'e'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(0, 8, 'f'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(0, 9, 'g'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(0, 10, 'h'));
   trainingData.add(LabeledFeatures.<Integer,Object>of(0, 11, 'i'));
   DecisionTree<Integer, Integer, Object> classifier = new DecisionTree<>(trainingData);
   //2nd feature selected - '1' chosen from '0,1' because 'a' always leads to '1'
   Assert.assertEquals(Integer.valueOf(1), classifier.classify(Features.<Object>of('x', 'a')));
    }
   
    @Test()
    public void myTest3() {
   List<LabeledFeatures<Integer, Integer, Object>> trainingData = new ArrayList<>();
   trainingData.add(LabeledFeatures.<Integer,Object>of(0, "sunny", 85, 85, false));
   trainingData.add(LabeledFeatures.<Integer,Object>of(0, "sunny", 80, 90, true));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, "overcast", 83, 78, false));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, "rain", 70, 96, false));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, "rain", 68, 80, false));
   trainingData.add(LabeledFeatures.<Integer,Object>of(0, "rain", 65, 70, true));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, "overcast", 64, 65, true));
   trainingData.add(LabeledFeatures.<Integer,Object>of(0, "sunny", 72, 95, false));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, "sunny", 69, 70, false));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, "rain", 75, 80, false));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, "sunny", 75, 70, true));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, "overcast", 72, 90, true));
   trainingData.add(LabeledFeatures.<Integer,Object>of(1, "overcast", 81, 75, false));
   trainingData.add(LabeledFeatures.<Integer,Object>of(0, "rain", 71, 80, true));
   DecisionTree<Integer, Integer, Object> classifier = new DecisionTree<>(trainingData);
   
   Assert.assertEquals(Integer.valueOf(0), classifier.classify(Features.<Object>of("rain", 85, 95, true)));

    }
      
   
    @Test()
    public void testFromSlide24ModifiedTrue() {
  //  >  x1 x2 f(x)
  //  >  0  a  true
  //  >  0  b  false
  //  >  1  c  true
  //  >  0  c  true
  //  >  0  c  false
  //  >  1  b  false
  //  >  1  a  true
   List<LabeledFeatures<Boolean, Integer, Object>> trainingData = new ArrayList<>();
   trainingData.add(LabeledFeatures.<Boolean,Object>of(true, 0, 'a'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(false, 0, 'b'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(true, 1, 'c'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(true, 0, 'c'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(false, 0, 'c'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(false, 1, 'b'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(true, 1, 'a'));
   DecisionTree<Boolean, Integer, Object> classifier = new DecisionTree<>(trainingData);
   
   Assert.assertEquals(true, classifier.classify(Features.<Object>of(2, 'c')));

    }
    
   
	/*
      @Test()
      public void testFromSlide24ModifiedFalse() {
  //  >  x1 x2 f(x)
  //  >  0  a  true
  //  >  0  b  false
  //  >  1  c  true
  //  >  0  c  false
  //  >  0  c  false
  //  >  1  b  false
  //  >  1  a  true
   List<LabeledFeatures<Boolean, Integer, Object>> trainingData = new ArrayList<>();
   trainingData.add(LabeledFeatures.<Boolean,Object>of(true, 0, 'a'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(false, 0, 'b'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(true, 1, 'c'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(false, 0, 'c'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(false, 0, 'c'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(false, 1, 'b'));
   trainingData.add(LabeledFeatures.<Boolean,Object>of(true, 1, 'a'));
   DecisionTree<Boolean, Integer, Object> classifier = new DecisionTree<>(trainingData);
   
   
   Assert.assertEquals(false, classifier.classify(Features.<Object>of(2, 'c')));

    }
   */  
    
      @Test()
  public void landonTest3() {
  List<LabeledFeatures<Object, Integer, Object>> trainingData = new ArrayList<>();
  trainingData.add(LabeledFeatures.<Object,Object>of("yellow", 5, 'a', 1));
  trainingData.add(LabeledFeatures.<Object,Object>of("yellow", 6, 'b', 1));
  trainingData.add(LabeledFeatures.<Object,Object>of("yellow", 7, 'c', 2));
  trainingData.add(LabeledFeatures.<Object,Object>of("yellow", 8, 'd', 3));
  trainingData.add(LabeledFeatures.<Object,Object>of("orange", 6, 'b', 4));
  trainingData.add(LabeledFeatures.<Object,Object>of("orange", 7, 'b', 2));
  trainingData.add(LabeledFeatures.<Object,Object>of("orange", 8, 'c', 3));
  trainingData.add(LabeledFeatures.<Object,Object>of("orange", 9, 'd', 2));
  trainingData.add(LabeledFeatures.<Object,Object>of("white", 8, 'd', 3));
  trainingData.add(LabeledFeatures.<Object,Object>of("white", 7, 'c', 3));
  trainingData.add(LabeledFeatures.<Object,Object>of("white", 8, 'd', 3));
  trainingData.add(LabeledFeatures.<Object,Object>of("white", 9, 'd', 4));

  DecisionTree<Object, Integer, Object> classifier = new DecisionTree<>(trainingData);
   

  Assert.assertEquals("yellow", classifier.classify(Features.<Object>of(10, "abc", 1)));
  Assert.assertEquals("orange", classifier.classify(Features.<Object>of(8, 'd', 2)));
  Assert.assertEquals("white", classifier.classify(Features.<Object>of(8, 'd', 3)));
  Assert.assertEquals("orange", classifier.classify(Features.<Object>of(8, 'c', 3)));
  Assert.assertEquals("orange", classifier.classify(Features.<Object>of(10, "abc", 2)));
  Assert.assertEquals("white", classifier.classify(Features.<Object>of(7, 'c', 3)));

  }
      

          @Test()
      public void landonTest2() {
   List<LabeledFeatures<Object, Integer, Object>> trainingData = new ArrayList<>();
   trainingData.add(LabeledFeatures.<Object,Object>of("green", 2, "dry", 5));
   trainingData.add(LabeledFeatures.<Object,Object>of("red", 3, "wet", 6));
   trainingData.add(LabeledFeatures.<Object,Object>of("blue", 3, "dry", 7));
   trainingData.add(LabeledFeatures.<Object,Object>of("green", 3, "wet", 8));
   trainingData.add(LabeledFeatures.<Object,Object>of("green", 2, "dry", 5));
   trainingData.add(LabeledFeatures.<Object,Object>of("red", 4, "wet", 6));
   trainingData.add(LabeledFeatures.<Object,Object>of("blue", 4, "dry", 7));
   trainingData.add(LabeledFeatures.<Object,Object>of("blue", 1, "wet", 8));
   trainingData.add(LabeledFeatures.<Object,Object>of("green", 1, "dry", 5));
   trainingData.add(LabeledFeatures.<Object,Object>of("red", 1, "dry", 8));
   trainingData.add(LabeledFeatures.<Object,Object>of("green", 1, "dry", 8));

   DecisionTree<Object, Integer, Object> classifier = new DecisionTree<>(trainingData);
   
   
   Assert.assertEquals("red", classifier.classify(Features.<Object>of(9, "abc", 6)));
   Assert.assertEquals("green", classifier.classify(Features.<Object>of(1, "wet", 5)));
   Assert.assertEquals("green", classifier.classify(Features.<Object>of(3, "dry")));
   Assert.assertEquals("blue", classifier.classify(Features.<Object>of(1, "wet", 8)));
   Assert.assertEquals("green", classifier.classify(Features.<Object>of(3, "wet", 8)));
   Assert.assertEquals("green", classifier.classify(Features.<Object>of(1, "wetwef", 8)));
   Assert.assertEquals("red", classifier.classify(Features.<Object>of(9, "dry", 8)));

    }

  
  // @formatter:off
  
  
}