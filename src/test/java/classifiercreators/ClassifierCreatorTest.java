package classifiercreators;

import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.Map;

import org.junit.Test;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;
import weka.tools.data.RandomDataGenerator;
import weka.tools.tests.DistributionChecker;

public class ClassifierCreatorTest {

	@Test
	public void testBaggingEnsembles() {
		RandomDataGenerator gen = new RandomDataGenerator();
		 gen.setNumNominalAttributes(0);
		 gen.setNumStringAttributes(0);
		 gen.setNumDateAttributes(0);
		 gen.setNumClasses(3);
		 gen.setNumObjects(300);
		 
		 Instances dataset = gen.generateData();
		 
		 Map<String,Classifier> ensemble = ClassifierCreator.createEnsemblesBagging(2);
		 for(Map.Entry<String, Classifier> entry : ensemble.entrySet()) {
			 String ensembleName = entry.getKey();
			 Classifier classifier = entry.getValue();
			 
			 try {
				classifier.buildClassifier(dataset);
				
				for(Instance instance: dataset) {
					
					double[] distribution = classifier.distributionForInstance(instance);
					assertTrue("Distribution, classifier: "+ensembleName, DistributionChecker.checkDistribution(distribution)); 
					
				}
				
			} catch (Exception e) {
				fail("Exception during classifier training:  " + e.getLocalizedMessage() );
				e.printStackTrace();
			}
			 
		 }
	}

	@Test
	public void testBoostingEnsembles() {
		RandomDataGenerator gen = new RandomDataGenerator();
		 gen.setNumNominalAttributes(0);
		 gen.setNumStringAttributes(0);
		 gen.setNumDateAttributes(0);
		 gen.setNumClasses(3);
		 gen.setNumObjects(300);
		 
		 Instances dataset = gen.generateData();
		 
		 Map<String,Classifier> ensemble = ClassifierCreator.createEnsemblesBoosting(2);
		 for(Map.Entry<String, Classifier> entry : ensemble.entrySet()) {
			 String ensembleName = entry.getKey();
			 Classifier classifier = entry.getValue();
			 
			 try {
				classifier.buildClassifier(dataset);
				
				for(Instance instance: dataset) {
					
					double[] distribution = classifier.distributionForInstance(instance);
					assertTrue("Distribution, classifier: "+ensembleName, DistributionChecker.checkDistribution(distribution)); 
					
				}
				
			} catch (Exception e) {
				fail("Exception during classifier training:  " + e.getLocalizedMessage() );
				e.printStackTrace();
			}
			 
		 }
		 
	}

}
