package classifiercreators;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.json.JSONArray;
import org.json.JSONObject;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.functions.BoundaryAndCentroidClassifier2;
import weka.classifiers.functions.BoundaryAndCentroidsClassifier;
import weka.classifiers.functions.BoundaryPotentialClassifier;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.explicitboundaries.ClassifierWithBoundaries;
import weka.classifiers.functions.explicitboundaries.combiners.PotentialFunctionCombiner;
import weka.classifiers.functions.explicitboundaries.combiners.PotentialFunctionLinear;
import weka.classifiers.functions.explicitboundaries.combiners.PotentialFunctionSign;
import weka.classifiers.functions.explicitboundaries.combiners.PotentialFunctionTanh;
import weka.classifiers.functions.explicitboundaries.combiners.potentialCombiners.PotentialCombinerSum;
import weka.classifiers.functions.explicitboundaries.models.FLDABoundary;
import weka.classifiers.functions.explicitboundaries.models.LibSVMSVCCLinearBoundary;
import weka.classifiers.functions.explicitboundaries.models.LogisticBoundary;
import weka.classifiers.functions.explicitboundaries.models.MultilayerPerceptronBoundary;
import weka.classifiers.functions.explicitboundaries.models.NearestCentroidBoundary;
import weka.classifiers.functions.explicitboundaries.models.SMOLinearBoundary;
import weka.classifiers.functions.nearestCentroid.prototypes.CustomizablePrototype;
import weka.classifiers.functions.nearestCentroid.prototypes.MahalanobisPrototype;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.CustomizableBaggingClassifier;
import weka.classifiers.meta.CustomizableBaggingClassifier2;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.meta.MultiClassClassifier;
import weka.classifiers.meta.MultiSearch;
import weka.classifiers.meta.MultiSearchZeroInstances;
import weka.classifiers.meta.RealAdaBoost;
import weka.classifiers.meta.generalOutputCombiners.VoteCombiner;
import weka.classifiers.meta.multisearch.DefaultEvaluationMetrics;
import weka.classifiers.meta.multisearch.DefaultSearch;
import weka.classifiers.meta.multisearch.SimpleCVSearchNoInstances;
import weka.classifiers.meta.simpleVotingLikeCombiners.BoundaryCombiner;
import weka.classifiers.trees.REPTree;
import weka.clusterers.ClassSpecificClusterer;
import weka.clusterers.XmeansWithKmeansPP;
import weka.core.SelectedTag;
import weka.core.UtilsPT;
import weka.core.distances.MahalanobisDistance;
import weka.core.setupgenerator.AbstractParameter;
import weka.core.setupgenerator.MathParameter;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.unsupervised.attribute.PrincipalComponents;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.Standardize;
import weka.tools.arrayFunctions.MaxFunction;
import weka.tools.arrayFunctions.MeanFunction;
import weka.tools.arrayFunctions.MultivariateFunction;
import weka.tools.data.splitters.PercentageSplitter;

public class ClassifierCreator {

	public static int seed=0;
	public static boolean debug = false;
	public static boolean noCheck = !debug;
	public static boolean lenient = true;

	public static int nThreads = 1;
	public static int initFolds=5;
	public static int subsequentFolds=10;

	public static AbstractClassifier createBaggingRT(Classifier baseClassifier, int committeeSize){

		FilteredClassifier filtered = new FilteredClassifier();
		MultiFilter multiFilter = new MultiFilter();
		multiFilter.setFilters(new Filter[]{new RemoveUseless()});
		filtered.setFilter(multiFilter);
		
		REPTree rt = new REPTree();
	
		CustomizableBaggingClassifier2 custBag = new CustomizableBaggingClassifier2();
		custBag.setClassifier(rt);
		custBag.setNumIterations(committeeSize);
		custBag.setSeed(seed);
		custBag.setDebug(debug);
		custBag.setDoNotCheckCapabilities(noCheck);
		custBag.setNumExecutionSlots(nThreads);
	
		VoteCombiner vComb = new VoteCombiner();
		
		custBag.setClassificationCombiner(vComb);
	
		filtered.setClassifier(custBag);
	
		return filtered;
	
	}

	public static AbstractClassifier createBaggingRT_PCA(Classifier baseClassifier, int committeeSize){

		FilteredClassifier filtered = new FilteredClassifier();
		MultiFilter multiFilter = new MultiFilter();
		multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents()});
		
		filtered.setFilter(multiFilter);
		
		REPTree rt = new REPTree();
	
		CustomizableBaggingClassifier2 custBag = new CustomizableBaggingClassifier2();
		custBag.setClassifier(rt);
		custBag.setNumIterations(committeeSize);
		custBag.setSeed(seed);
		custBag.setDebug(debug);
		custBag.setDoNotCheckCapabilities(noCheck);
		custBag.setNumExecutionSlots(nThreads);
	
		VoteCombiner vComb = new VoteCombiner();
		
		custBag.setClassificationCombiner(vComb);
	
		filtered.setClassifier(custBag);
	
		return filtered;
	
	}


	public static AbstractClassifier createBaggingRTM(Classifier baseClassifier, int committeeSize){

		FilteredClassifier filtered = new FilteredClassifier();
		MultiFilter multiFilter = new MultiFilter();
		// multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents(), new Standardize()});
		// multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents()});
		multiFilter.setFilters(new Filter[]{new RemoveUseless()});
		filtered.setFilter(multiFilter);

		MultiClassClassifier multiClass1 = new MultiClassClassifier();
		multiClass1.setMethod(new SelectedTag(MultiClassClassifier.METHOD_1_AGAINST_1, MultiClassClassifier.TAGS_METHOD));
		
		REPTree rt = new REPTree();
	
		CustomizableBaggingClassifier2 custBag = new CustomizableBaggingClassifier2();
		custBag.setClassifier(rt);
		custBag.setNumIterations(committeeSize);
		custBag.setSeed(seed);
		custBag.setDebug(debug);
		custBag.setDoNotCheckCapabilities(noCheck);
		custBag.setNumExecutionSlots(nThreads);
	
		VoteCombiner vComb = new VoteCombiner();
		
		custBag.setClassificationCombiner(vComb);
	
		multiClass1.setClassifier(custBag);
		filtered.setClassifier(multiClass1);
	
		return filtered;
	
	}

public static AbstractClassifier createBaggingRTT(Classifier baseClassifier, int committeeSize){

		FilteredClassifier filtered = new FilteredClassifier();
		MultiFilter multiFilter = new MultiFilter();
		// multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents(), new Standardize()});
		// multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents()});
		multiFilter.setFilters(new Filter[]{new RemoveUseless()});
		filtered.setFilter(multiFilter);
	
		REPTree rt = new REPTree();
	
		CustomizableBaggingClassifier custBag = new CustomizableBaggingClassifier();
		custBag.setClassifier(rt);
		custBag.setNumIterations(committeeSize);
		custBag.setSeed(seed);
		custBag.setDebug(debug);
		custBag.setDoNotCheckCapabilities(noCheck);
		custBag.setNumExecutionSlots(nThreads);
	
	
		MathParameter mParam = new MathParameter();
		mParam.setProperty("Classifier.MaxDepth");
		mParam.setMin(-1);
		mParam.setMax(10);
		mParam.setStep(1);
	
				
				
		DefaultSearch defSearch = new SimpleCVSearchNoInstances();
		defSearch.setDebug(debug);
		defSearch.setInitialSpaceNumFolds(initFolds);
		defSearch.setSubsequentSpaceNumFolds(subsequentFolds);
		defSearch.setLenient(lenient);
	
		MultiSearch mSearch = new MultiSearchZeroInstances();
		mSearch.setClassifier(custBag);
		mSearch.setSearchParameters(new AbstractParameter[]{mParam});
		DefaultEvaluationMetrics evMetr = new DefaultEvaluationMetrics();
		mSearch.setEvaluation(new SelectedTag(DefaultEvaluationMetrics.EVALUATION_KAPPA, evMetr.getTags()));
		mSearch.setDebug(debug);
		mSearch.setSeed(seed);
		mSearch.setAlgorithm(defSearch);
	
		filtered.setClassifier(mSearch);
	
		return filtered;
	}

public static AbstractClassifier createBaggingRTTM(Classifier baseClassifier, int committeeSize){

		FilteredClassifier filtered = new FilteredClassifier();
		MultiFilter multiFilter = new MultiFilter();
		// multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents(), new Standardize()});
		// multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents()});
		multiFilter.setFilters(new Filter[]{new RemoveUseless()});
		filtered.setFilter(multiFilter);

		MultiClassClassifier multiClass1 = new MultiClassClassifier();
		multiClass1.setMethod(new SelectedTag(MultiClassClassifier.METHOD_1_AGAINST_1, MultiClassClassifier.TAGS_METHOD));
	
		REPTree rt = new REPTree();
	
		CustomizableBaggingClassifier custBag = new CustomizableBaggingClassifier();
		custBag.setClassifier(rt);
		custBag.setNumIterations(committeeSize);
		custBag.setSeed(seed);
		custBag.setDebug(debug);
		custBag.setDoNotCheckCapabilities(noCheck);
		custBag.setNumExecutionSlots(nThreads);
	
	
		MathParameter mParam = new MathParameter();
		mParam.setProperty("Classifier.MaxDepth");
		mParam.setMin(-1);
		mParam.setMax(10);
		mParam.setStep(1);
	
				
				
		DefaultSearch defSearch = new SimpleCVSearchNoInstances();
		defSearch.setDebug(debug);
		defSearch.setInitialSpaceNumFolds(initFolds);
		defSearch.setSubsequentSpaceNumFolds(subsequentFolds);
		defSearch.setLenient(lenient);
	
		MultiSearch mSearch = new MultiSearchZeroInstances();
		mSearch.setClassifier(custBag);
		mSearch.setSearchParameters(new AbstractParameter[]{mParam});
		DefaultEvaluationMetrics evMetr = new DefaultEvaluationMetrics();
		mSearch.setEvaluation(new SelectedTag(DefaultEvaluationMetrics.EVALUATION_KAPPA, evMetr.getTags()));
		mSearch.setDebug(debug);
		mSearch.setSeed(seed);
		mSearch.setAlgorithm(defSearch);
	
		multiClass1.setClassifier(custBag);
		filtered.setClassifier(multiClass1);
	
		return filtered;
	}



public static AbstractClassifier createBaggingMV(Classifier baseClassifier, int committeeSize){

	FilteredClassifier filtered = new FilteredClassifier();
	MultiFilter multiFilter = new MultiFilter();
	multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents(), new Standardize()});
	filtered.setFilter(multiFilter);
		
	MultiClassClassifier multiClass1 = new MultiClassClassifier();
	multiClass1.setMethod(new SelectedTag(MultiClassClassifier.METHOD_1_AGAINST_1, MultiClassClassifier.TAGS_METHOD));

	CustomizableBaggingClassifier2 custBag = new CustomizableBaggingClassifier2();
	custBag.setClassifier(baseClassifier);
	custBag.setNumIterations(committeeSize);
	custBag.setSeed(seed);
	custBag.setDebug(debug);
	custBag.setDoNotCheckCapabilities(noCheck);
	custBag.setNumExecutionSlots(nThreads);

	VoteCombiner vComb = new VoteCombiner();
	
	custBag.setClassificationCombiner(vComb);

	multiClass1.setClassifier(custBag);
	filtered.setClassifier(multiClass1);

	return filtered;

}

public static AbstractClassifier createBaggingMA(Classifier baseClassifier, int committeeSize){

	FilteredClassifier filtered = new FilteredClassifier();
	MultiFilter multiFilter = new MultiFilter();
	multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents(), new Standardize()});
	filtered.setFilter(multiFilter);
		
	MultiClassClassifier multiClass1 = new MultiClassClassifier();
	multiClass1.setMethod(new SelectedTag(MultiClassClassifier.METHOD_1_AGAINST_1, MultiClassClassifier.TAGS_METHOD));

	CustomizableBaggingClassifier custBag = new CustomizableBaggingClassifier();
	custBag.setClassifier(baseClassifier);
	custBag.setNumIterations(committeeSize);
	custBag.setSeed(seed);
	custBag.setDebug(debug);
	custBag.setDoNotCheckCapabilities(noCheck);
	custBag.setNumExecutionSlots(nThreads);

	BoundaryCombiner boundCombiner = new BoundaryCombiner();
	PotentialFunctionCombiner boundaryCombiner = new PotentialFunctionCombiner();
	boundaryCombiner.setPotential(new PotentialFunctionLinear());
	boundaryCombiner.setPotCombiner(new PotentialCombinerSum());
	boundCombiner.setBoundaryCombiner(boundaryCombiner );

	custBag.setOutCombiner(boundCombiner);

	multiClass1.setClassifier(custBag);
	filtered.setClassifier(multiClass1);

	return filtered;

}

public static AbstractClassifier createBaggingTanhBest(Classifier baseClassifier, int committeeSize){

	FilteredClassifier filtered = new FilteredClassifier();
	MultiFilter multiFilter = new MultiFilter();
	multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents(), new Standardize()});
	filtered.setFilter(multiFilter);
		
	MultiClassClassifier multiClass1 = new MultiClassClassifier();
	multiClass1.setMethod(new SelectedTag(MultiClassClassifier.METHOD_1_AGAINST_1, MultiClassClassifier.TAGS_METHOD));

	CustomizableBaggingClassifier custBag = new CustomizableBaggingClassifier();
	custBag.setClassifier(baseClassifier);
	custBag.setNumIterations(committeeSize);
	custBag.setSeed(seed);
	custBag.setDebug(debug);
	custBag.setDoNotCheckCapabilities(noCheck);
	custBag.setNumExecutionSlots(nThreads);

	BoundaryCombiner boundCombiner = new BoundaryCombiner();
	PotentialFunctionCombiner boundaryCombiner = new PotentialFunctionCombiner();
	boundaryCombiner.setPotential(new PotentialFunctionTanh());
	boundaryCombiner.setPotCombiner(new PotentialCombinerSum());
	boundCombiner.setBoundaryCombiner(boundaryCombiner );

	custBag.setOutCombiner(boundCombiner);

	MathParameter mParam = new MathParameter();
	mParam.setProperty("outCombiner.boundaryCombiner.potential.Alpha");
	mParam.setMin(-4);
	mParam.setMax(4);
	mParam.setStep(1);
	mParam.setBase(2);
			
			
	DefaultSearch defSearch = new SimpleCVSearchNoInstances();
	defSearch.setDebug(debug);
	defSearch.setInitialSpaceNumFolds(initFolds);
	defSearch.setSubsequentSpaceNumFolds(subsequentFolds);
	defSearch.setLenient(lenient);

	MultiSearch mSearch = new MultiSearchZeroInstances();
	mSearch.setClassifier(custBag);
	mSearch.setSearchParameters(new AbstractParameter[]{mParam});
	DefaultEvaluationMetrics evMetr = new DefaultEvaluationMetrics();
	mSearch.setEvaluation(new SelectedTag(DefaultEvaluationMetrics.EVALUATION_KAPPA, evMetr.getTags()));
	mSearch.setDebug(debug);
	mSearch.setSeed(seed);
	mSearch.setAlgorithm(defSearch);


	multiClass1.setClassifier(mSearch);
	filtered.setClassifier(multiClass1);

	return filtered;
}



public static AbstractClassifier createBaggingBoundAndCentroid(Classifier baseClassifier, int committeeSize){

	FilteredClassifier filtered = new FilteredClassifier();
	MultiFilter multiFilter = new MultiFilter();
	multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents(), new Standardize()});
	filtered.setFilter(multiFilter);
		
	MultiClassClassifier multiClass1 = new MultiClassClassifier();
	multiClass1.setMethod(new SelectedTag(MultiClassClassifier.METHOD_1_AGAINST_1, MultiClassClassifier.TAGS_METHOD));


	BoundaryAndCentroidClassifier2 bndAndCent = new BoundaryAndCentroidClassifier2((ClassifierWithBoundaries) baseClassifier);

	CustomizableBaggingClassifier custBag = new CustomizableBaggingClassifier();
	custBag.setClassifier(bndAndCent);
	custBag.setNumIterations(committeeSize);
	custBag.setSeed(seed);
	custBag.setDebug(debug);
	custBag.setDoNotCheckCapabilities(noCheck);
	custBag.setNumExecutionSlots(nThreads);


	MathParameter mParam = new MathParameter();
	mParam.setProperty("Classifier.proportion");
	mParam.setMin(0);
	mParam.setMax(1.0);
	mParam.setStep(0.1);
	mParam.setExpression("I");

			
			
	DefaultSearch defSearch = new SimpleCVSearchNoInstances();
	defSearch.setDebug(debug);
	defSearch.setInitialSpaceNumFolds(initFolds);
	defSearch.setSubsequentSpaceNumFolds(subsequentFolds);
	defSearch.setLenient(lenient);

	MultiSearch mSearch = new MultiSearchZeroInstances();
	mSearch.setClassifier(custBag);
	mSearch.setSearchParameters(new AbstractParameter[]{mParam});
	DefaultEvaluationMetrics evMetr = new DefaultEvaluationMetrics();
	mSearch.setEvaluation(new SelectedTag(DefaultEvaluationMetrics.EVALUATION_KAPPA, evMetr.getTags()));
	mSearch.setDebug(debug);
	mSearch.setSeed(seed);
	mSearch.setAlgorithm(defSearch);


	multiClass1.setClassifier(mSearch);
	filtered.setClassifier(multiClass1);

	return filtered;
}


public static AbstractClassifier createBaggingBoundAndCentroids(Classifier baseClassifier, int committeeSize){

	FilteredClassifier filtered = new FilteredClassifier();
	MultiFilter multiFilter = new MultiFilter();
	multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents(), new Standardize()});
	filtered.setFilter(multiFilter);
		
	MultiClassClassifier multiClass1 = new MultiClassClassifier();
	multiClass1.setMethod(new SelectedTag(MultiClassClassifier.METHOD_1_AGAINST_1, MultiClassClassifier.TAGS_METHOD));


	BoundaryAndCentroidsClassifier bndAndCent = new BoundaryAndCentroidsClassifier((ClassifierWithBoundaries) baseClassifier) ;
	
	MultivariateFunction clusterCombiner = new MeanFunction();
	bndAndCent.setClusterCombiner(clusterCombiner);
	bndAndCent.setPrototypeProto(new MahalanobisPrototype());

	PercentageSplitter splitter = new PercentageSplitter();
	bndAndCent.setDataSplitter(splitter);

	ClassSpecificClusterer clSpecClusterer = new ClassSpecificClusterer();
	XmeansWithKmeansPP xmeans = new XmeansWithKmeansPP();
	xmeans.setMinNumClusters(2);
	xmeans.setMaxNumClusters(5);
	xmeans.setDistanceF(new MahalanobisDistance());
	clSpecClusterer.setClusterer(xmeans);
	bndAndCent.setClassSpecificClusterer(clSpecClusterer);

	CustomizableBaggingClassifier custBag = new CustomizableBaggingClassifier();
	custBag.setClassifier(bndAndCent);
	custBag.setNumIterations(committeeSize);
	custBag.setSeed(seed);
	custBag.setDebug(debug);
	custBag.setDoNotCheckCapabilities(noCheck);
	custBag.setNumExecutionSlots(nThreads);


	MathParameter mParam = new MathParameter();
	mParam.setProperty("Classifier.proportion");
	mParam.setMin(0);
	mParam.setMax(1.0);
	mParam.setStep(0.1);
	mParam.setExpression("I");

			
			
	DefaultSearch defSearch = new SimpleCVSearchNoInstances();
	defSearch.setDebug(debug);
	defSearch.setInitialSpaceNumFolds(initFolds);
	defSearch.setSubsequentSpaceNumFolds(subsequentFolds);
	defSearch.setLenient(lenient);

	MultiSearch mSearch = new MultiSearchZeroInstances();
	mSearch.setClassifier(custBag);
	mSearch.setSearchParameters(new AbstractParameter[]{mParam});
	DefaultEvaluationMetrics evMetr = new DefaultEvaluationMetrics();
	mSearch.setEvaluation(new SelectedTag(DefaultEvaluationMetrics.EVALUATION_KAPPA, evMetr.getTags()));
	mSearch.setDebug(debug);
	mSearch.setSeed(seed);
	mSearch.setAlgorithm(defSearch);


	multiClass1.setClassifier(mSearch);
	filtered.setClassifier(multiClass1);

	return filtered;
}


public static AbstractClassifier createBaggingBoundAndCentroids2(Classifier baseClassifier, int committeeSize){

	FilteredClassifier filtered = new FilteredClassifier();
	MultiFilter multiFilter = new MultiFilter();
	multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents(), new Standardize()});
	filtered.setFilter(multiFilter);
		
	MultiClassClassifier multiClass1 = new MultiClassClassifier();
	multiClass1.setMethod(new SelectedTag(MultiClassClassifier.METHOD_1_AGAINST_1, MultiClassClassifier.TAGS_METHOD));


	BoundaryAndCentroidsClassifier bndAndCent = new BoundaryAndCentroidsClassifier((ClassifierWithBoundaries) baseClassifier) ;
	bndAndCent.setPrototypeProto(new MahalanobisPrototype());
	MultivariateFunction clusterCombiner = new MaxFunction();
	bndAndCent.setClusterCombiner(clusterCombiner);

	PercentageSplitter splitter = new PercentageSplitter();
	bndAndCent.setDataSplitter(splitter);

	ClassSpecificClusterer clSpecClusterer = new ClassSpecificClusterer();
	XmeansWithKmeansPP xmeans = new XmeansWithKmeansPP();
	xmeans.setMinNumClusters(2);
	xmeans.setMaxNumClusters(5);
	xmeans.setDistanceF(new MahalanobisDistance());
	clSpecClusterer.setClusterer(xmeans);
	bndAndCent.setClassSpecificClusterer(clSpecClusterer);

	CustomizableBaggingClassifier custBag = new CustomizableBaggingClassifier();
	custBag.setClassifier(bndAndCent);
	custBag.setNumIterations(committeeSize);
	custBag.setSeed(seed);
	custBag.setDebug(debug);
	custBag.setDoNotCheckCapabilities(noCheck);
	custBag.setNumExecutionSlots(nThreads);


	MathParameter mParam = new MathParameter();
	mParam.setProperty("Classifier.proportion");
	mParam.setMin(0);
	mParam.setMax(1.0);
	mParam.setStep(0.1);
	mParam.setExpression("I");

			
			
	DefaultSearch defSearch = new SimpleCVSearchNoInstances();
	defSearch.setDebug(debug);
	defSearch.setInitialSpaceNumFolds(initFolds);
	defSearch.setSubsequentSpaceNumFolds(subsequentFolds);
	defSearch.setLenient(lenient);

	MultiSearch mSearch = new MultiSearchZeroInstances();
	mSearch.setClassifier(custBag);
	mSearch.setSearchParameters(new AbstractParameter[]{mParam});
	DefaultEvaluationMetrics evMetr = new DefaultEvaluationMetrics();
	mSearch.setEvaluation(new SelectedTag(DefaultEvaluationMetrics.EVALUATION_KAPPA, evMetr.getTags()));
	mSearch.setDebug(debug);
	mSearch.setSeed(seed);
	mSearch.setAlgorithm(defSearch);


	multiClass1.setClassifier(mSearch);
	filtered.setClassifier(multiClass1);

	return filtered;
}

public static AbstractClassifier createBoostingSign(Classifier baseClassifier, int committeeSize){

	FilteredClassifier filtered = new FilteredClassifier();
	MultiFilter multiFilter = new MultiFilter();
	multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents(), new Standardize()});
	filtered.setFilter(multiFilter);
		
	MultiClassClassifier multiClass1 = new MultiClassClassifier();
	multiClass1.setMethod(new SelectedTag(MultiClassClassifier.METHOD_1_AGAINST_1, MultiClassClassifier.TAGS_METHOD));


	BoundaryPotentialClassifier bPotClassifier =  new BoundaryPotentialClassifier((ClassifierWithBoundaries) baseClassifier);
	bPotClassifier.setPotential( new PotentialFunctionSign());

	RealAdaBoost boost = new RealAdaBoost();
	boost.setSeed(seed);
	boost.setDebug(debug);
	boost.setDoNotCheckCapabilities(noCheck);
	boost.setClassifier(bPotClassifier);
	boost.setNumIterations(committeeSize);

	multiClass1.setClassifier(boost);
	filtered.setClassifier(multiClass1);

	return filtered;

}

public static AbstractClassifier createBoostingRT(Classifier baseClassifier, int committeeSize){

	FilteredClassifier filtered = new FilteredClassifier();
	MultiFilter multiFilter = new MultiFilter();
	multiFilter.setFilters(new Filter[]{new RemoveUseless()});
	filtered.setFilter(multiFilter);

	BoundaryPotentialClassifier bPotClassifier =  new BoundaryPotentialClassifier((ClassifierWithBoundaries) baseClassifier);
	bPotClassifier.setPotential( new PotentialFunctionTanh());

	REPTree rt = new REPTree();

	//Boosting
	AdaBoostM1 boost = new AdaBoostM1();
	boost.setSeed(seed);
	boost.setDebug(debug);
	boost.setDoNotCheckCapabilities(noCheck);
	boost.setClassifier(rt);
	boost.setNumIterations(committeeSize);


	MathParameter mParam = new MathParameter();
	mParam.setProperty("Classifier.MaxDepth");
	mParam.setMin(1);
	mParam.setMax(10);
	mParam.setStep(1);
			
			
	DefaultSearch defSearch = new SimpleCVSearchNoInstances();
	defSearch.setDebug(debug);
	defSearch.setInitialSpaceNumFolds(initFolds);
	defSearch.setSubsequentSpaceNumFolds(subsequentFolds);
	defSearch.setLenient(lenient);

	MultiSearch mSearch = new MultiSearchZeroInstances();
	mSearch.setClassifier(boost);
	mSearch.setSearchParameters(new AbstractParameter[]{mParam});
	DefaultEvaluationMetrics evMetr = new DefaultEvaluationMetrics();
	mSearch.setEvaluation(new SelectedTag(DefaultEvaluationMetrics.EVALUATION_KAPPA, evMetr.getTags()));
	mSearch.setDebug(debug);
	mSearch.setSeed(seed);
	mSearch.setAlgorithm(defSearch);

	
	filtered.setClassifier(mSearch);

	return filtered;
}

public static AbstractClassifier createBoostingRT_PCA(Classifier baseClassifier, int committeeSize){

	FilteredClassifier filtered = new FilteredClassifier();
	MultiFilter multiFilter = new MultiFilter();
	multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents()});
	
	filtered.setFilter(multiFilter);

	BoundaryPotentialClassifier bPotClassifier =  new BoundaryPotentialClassifier((ClassifierWithBoundaries) baseClassifier);
	bPotClassifier.setPotential( new PotentialFunctionTanh());

	REPTree rt = new REPTree();

	//Boosting
	AdaBoostM1 boost = new AdaBoostM1();
	boost.setSeed(seed);
	boost.setDebug(debug);
	boost.setDoNotCheckCapabilities(noCheck);
	boost.setClassifier(rt);
	boost.setNumIterations(committeeSize);


	MathParameter mParam = new MathParameter();
	mParam.setProperty("Classifier.MaxDepth");
	mParam.setMin(1);
	mParam.setMax(10);
	mParam.setStep(1);
			
			
	DefaultSearch defSearch = new SimpleCVSearchNoInstances();
	defSearch.setDebug(debug);
	defSearch.setInitialSpaceNumFolds(initFolds);
	defSearch.setSubsequentSpaceNumFolds(subsequentFolds);
	defSearch.setLenient(lenient);

	MultiSearch mSearch = new MultiSearchZeroInstances();
	mSearch.setClassifier(boost);
	mSearch.setSearchParameters(new AbstractParameter[]{mParam});
	DefaultEvaluationMetrics evMetr = new DefaultEvaluationMetrics();
	mSearch.setEvaluation(new SelectedTag(DefaultEvaluationMetrics.EVALUATION_KAPPA, evMetr.getTags()));
	mSearch.setDebug(debug);
	mSearch.setSeed(seed);
	mSearch.setAlgorithm(defSearch);

	
	filtered.setClassifier(mSearch);

	return filtered;
}

public static AbstractClassifier createBoostingRTM(Classifier baseClassifier, int committeeSize){

	FilteredClassifier filtered = new FilteredClassifier();
	MultiFilter multiFilter = new MultiFilter();

	// multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents()});
	multiFilter.setFilters(new Filter[]{new RemoveUseless()});
	filtered.setFilter(multiFilter);

	MultiClassClassifier multiClass1 = new MultiClassClassifier();
	multiClass1.setMethod(new SelectedTag(MultiClassClassifier.METHOD_1_AGAINST_1, MultiClassClassifier.TAGS_METHOD));

	BoundaryPotentialClassifier bPotClassifier =  new BoundaryPotentialClassifier((ClassifierWithBoundaries) baseClassifier);
	bPotClassifier.setPotential( new PotentialFunctionTanh());

	REPTree rt = new REPTree();

	//Boosting
	AdaBoostM1 boost = new AdaBoostM1();
	boost.setSeed(seed);
	boost.setDebug(debug);
	boost.setDoNotCheckCapabilities(noCheck);
	boost.setClassifier(rt);
	boost.setNumIterations(committeeSize);


	MathParameter mParam = new MathParameter();
	mParam.setProperty("Classifier.MaxDepth");
	mParam.setMin(1);
	mParam.setMax(10);
	mParam.setStep(1);
			
			
	DefaultSearch defSearch = new SimpleCVSearchNoInstances();
	defSearch.setDebug(debug);
	defSearch.setInitialSpaceNumFolds(initFolds);
	defSearch.setSubsequentSpaceNumFolds(subsequentFolds);
	defSearch.setLenient(lenient);

	MultiSearch mSearch = new MultiSearchZeroInstances();
	mSearch.setClassifier(boost);
	mSearch.setSearchParameters(new AbstractParameter[]{mParam});
	DefaultEvaluationMetrics evMetr = new DefaultEvaluationMetrics();
	mSearch.setEvaluation(new SelectedTag(DefaultEvaluationMetrics.EVALUATION_KAPPA, evMetr.getTags()));
	mSearch.setDebug(debug);
	mSearch.setSeed(seed);
	mSearch.setAlgorithm(defSearch);

	
	multiClass1.setClassifier(boost);
	filtered.setClassifier(multiClass1);

	return filtered;
}


public static AbstractClassifier createBoostingTanhBest(Classifier baseClassifier, int committeeSize){

	FilteredClassifier filtered = new FilteredClassifier();
	MultiFilter multiFilter = new MultiFilter();
	multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents(), new Standardize()});
	filtered.setFilter(multiFilter);
		
	MultiClassClassifier multiClass1 = new MultiClassClassifier();
	multiClass1.setMethod(new SelectedTag(MultiClassClassifier.METHOD_1_AGAINST_1, MultiClassClassifier.TAGS_METHOD));

	BoundaryPotentialClassifier bPotClassifier =  new BoundaryPotentialClassifier((ClassifierWithBoundaries) baseClassifier);
	bPotClassifier.setPotential( new PotentialFunctionTanh());

	//Boosting
	RealAdaBoost boost = new RealAdaBoost();
	boost.setSeed(seed);
	boost.setDebug(debug);
	boost.setDoNotCheckCapabilities(noCheck);
	boost.setClassifier(bPotClassifier);
	boost.setNumIterations(committeeSize);


	MathParameter mParam = new MathParameter();
	mParam.setProperty("Classifier.potential.alpha");
	mParam.setMin(-4);
	mParam.setMax(4);
	mParam.setStep(1);
	mParam.setBase(2);
			
			
	DefaultSearch defSearch = new SimpleCVSearchNoInstances();
	defSearch.setDebug(debug);
	defSearch.setInitialSpaceNumFolds(initFolds);
	defSearch.setSubsequentSpaceNumFolds(subsequentFolds);
	defSearch.setLenient(lenient);

	MultiSearch mSearch = new MultiSearchZeroInstances();
	mSearch.setClassifier(boost);
	mSearch.setSearchParameters(new AbstractParameter[]{mParam});
	DefaultEvaluationMetrics evMetr = new DefaultEvaluationMetrics();
	mSearch.setEvaluation(new SelectedTag(DefaultEvaluationMetrics.EVALUATION_KAPPA, evMetr.getTags()));
	mSearch.setDebug(debug);
	mSearch.setSeed(seed);
	mSearch.setAlgorithm(defSearch);

	

	multiClass1.setClassifier(mSearch);
	filtered.setClassifier(multiClass1);

	return filtered;
}

public static AbstractClassifier createBoostingBoundAndCentroid(Classifier baseClassifier, int committeeSize){

	FilteredClassifier filtered = new FilteredClassifier();
	MultiFilter multiFilter = new MultiFilter();
	multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents(), new Standardize()});
	filtered.setFilter(multiFilter);
		
	MultiClassClassifier multiClass1 = new MultiClassClassifier();
	multiClass1.setMethod(new SelectedTag(MultiClassClassifier.METHOD_1_AGAINST_1, MultiClassClassifier.TAGS_METHOD));


	BoundaryAndCentroidClassifier2 bndAndCent = new BoundaryAndCentroidClassifier2((ClassifierWithBoundaries) baseClassifier);

	//Boosting
	RealAdaBoost boost = new RealAdaBoost();
	boost.setSeed(seed);
	boost.setDebug(debug);
	boost.setDoNotCheckCapabilities(noCheck);
	boost.setClassifier(bndAndCent);
	boost.setNumIterations(committeeSize);



	MathParameter mParam = new MathParameter();
	mParam.setProperty("Classifier.proportion");
	mParam.setMin(0);
	mParam.setMax(1.0);
	mParam.setStep(0.1);
	mParam.setExpression("I");

			
			
	DefaultSearch defSearch = new SimpleCVSearchNoInstances();
	defSearch.setDebug(debug);
	defSearch.setInitialSpaceNumFolds(initFolds);
	defSearch.setSubsequentSpaceNumFolds(subsequentFolds);
	defSearch.setLenient(lenient);

	MultiSearch mSearch = new MultiSearchZeroInstances();
	mSearch.setClassifier(boost);
	mSearch.setSearchParameters(new AbstractParameter[]{mParam});
	DefaultEvaluationMetrics evMetr = new DefaultEvaluationMetrics();
	mSearch.setEvaluation(new SelectedTag(DefaultEvaluationMetrics.EVALUATION_KAPPA, evMetr.getTags()));
	mSearch.setDebug(debug);
	mSearch.setSeed(seed);
	mSearch.setAlgorithm(defSearch);

	

	multiClass1.setClassifier(mSearch);
	filtered.setClassifier(multiClass1);

	return filtered;
}

public static AbstractClassifier createBoostingBoundAndCentroids(Classifier baseClassifier, int committeeSize){

	FilteredClassifier filtered = new FilteredClassifier();
	MultiFilter multiFilter = new MultiFilter();
	multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents(), new Standardize()});
	filtered.setFilter(multiFilter);
		
	MultiClassClassifier multiClass1 = new MultiClassClassifier();
	multiClass1.setMethod(new SelectedTag(MultiClassClassifier.METHOD_1_AGAINST_1, MultiClassClassifier.TAGS_METHOD));


	BoundaryAndCentroidsClassifier bndAndCent = new BoundaryAndCentroidsClassifier((ClassifierWithBoundaries) baseClassifier) ;
	
	MultivariateFunction clusterCombiner = new MeanFunction();
	bndAndCent.setClusterCombiner(clusterCombiner);
	bndAndCent.setPrototypeProto(new MahalanobisPrototype());

	PercentageSplitter splitter = new PercentageSplitter();
	bndAndCent.setDataSplitter(splitter);

	ClassSpecificClusterer clSpecClusterer = new ClassSpecificClusterer();
	XmeansWithKmeansPP xmeans = new XmeansWithKmeansPP();
	xmeans.setMinNumClusters(2);
	xmeans.setMaxNumClusters(5);
	xmeans.setDistanceF(new MahalanobisDistance());
	clSpecClusterer.setClusterer(xmeans);
	bndAndCent.setClassSpecificClusterer(clSpecClusterer);


	//Boosting
	RealAdaBoost boost = new RealAdaBoost();
	boost.setSeed(seed);
	boost.setDebug(debug);
	boost.setDoNotCheckCapabilities(noCheck);
	boost.setClassifier(bndAndCent);
	boost.setNumIterations(committeeSize);


	MathParameter mParam = new MathParameter();
	mParam.setProperty("Classifier.proportion");
	mParam.setMin(0);
	mParam.setMax(1.0);
	mParam.setStep(0.1);
	mParam.setExpression("I");

			
			
	DefaultSearch defSearch = new SimpleCVSearchNoInstances();
	defSearch.setDebug(debug);
	defSearch.setInitialSpaceNumFolds(initFolds);
	defSearch.setSubsequentSpaceNumFolds(subsequentFolds);
	defSearch.setLenient(lenient);

	MultiSearch mSearch = new MultiSearchZeroInstances();
	mSearch.setClassifier(boost);
	mSearch.setSearchParameters(new AbstractParameter[]{mParam});
	DefaultEvaluationMetrics evMetr = new DefaultEvaluationMetrics();
	mSearch.setEvaluation(new SelectedTag(DefaultEvaluationMetrics.EVALUATION_KAPPA, evMetr.getTags()));
	mSearch.setDebug(debug);
	mSearch.setSeed(seed);
	mSearch.setAlgorithm(defSearch);



	multiClass1.setClassifier(mSearch);
	filtered.setClassifier(multiClass1);

	return filtered;
}

public static AbstractClassifier createBoostingBoundAndCentroids2(Classifier baseClassifier, int committeeSize){

	FilteredClassifier filtered = new FilteredClassifier();
	MultiFilter multiFilter = new MultiFilter();
	multiFilter.setFilters(new Filter[]{new RemoveUseless(), new PrincipalComponents(), new Standardize()});
	filtered.setFilter(multiFilter);
		
	MultiClassClassifier multiClass1 = new MultiClassClassifier();
	multiClass1.setMethod(new SelectedTag(MultiClassClassifier.METHOD_1_AGAINST_1, MultiClassClassifier.TAGS_METHOD));


	BoundaryAndCentroidsClassifier bndAndCent = new BoundaryAndCentroidsClassifier((ClassifierWithBoundaries) baseClassifier) ;
	bndAndCent.setPrototypeProto(new MahalanobisPrototype());
	MultivariateFunction clusterCombiner = new MaxFunction();
	bndAndCent.setClusterCombiner(clusterCombiner);

	PercentageSplitter splitter = new PercentageSplitter();
	bndAndCent.setDataSplitter(splitter);

	ClassSpecificClusterer clSpecClusterer = new ClassSpecificClusterer();
	XmeansWithKmeansPP xmeans = new XmeansWithKmeansPP();
	xmeans.setMinNumClusters(2);
	xmeans.setMaxNumClusters(5);
	xmeans.setDistanceF(new MahalanobisDistance());
	clSpecClusterer.setClusterer(xmeans);
	bndAndCent.setClassSpecificClusterer(clSpecClusterer);


	//Boosting
	RealAdaBoost boost = new RealAdaBoost();
	boost.setSeed(seed);
	boost.setDebug(debug);
	boost.setDoNotCheckCapabilities(noCheck);
	boost.setClassifier(bndAndCent);
	boost.setNumIterations(committeeSize);


	MathParameter mParam = new MathParameter();
	mParam.setProperty("Classifier.proportion");
	mParam.setMin(0);
	mParam.setMax(1.0);
	mParam.setStep(0.1);
	mParam.setExpression("I");

			
			
	DefaultSearch defSearch = new SimpleCVSearchNoInstances();
	defSearch.setDebug(debug);
	defSearch.setInitialSpaceNumFolds(initFolds);
	defSearch.setSubsequentSpaceNumFolds(subsequentFolds);
	defSearch.setLenient(lenient);

	MultiSearch mSearch = new MultiSearchZeroInstances();
	mSearch.setClassifier(boost);
	mSearch.setSearchParameters(new AbstractParameter[]{mParam});
	DefaultEvaluationMetrics evMetr = new DefaultEvaluationMetrics();
	mSearch.setEvaluation(new SelectedTag(DefaultEvaluationMetrics.EVALUATION_KAPPA, evMetr.getTags()));
	mSearch.setDebug(debug);
	mSearch.setSeed(seed);
	mSearch.setAlgorithm(defSearch);


	multiClass1.setClassifier(mSearch);
	filtered.setClassifier(multiClass1);

	return filtered;
}




	public static Classifier createNCBound() {
		
		NearestCentroidBoundary bnd = new NearestCentroidBoundary();
		bnd.setClusProto(new CustomizablePrototype());

		bnd.setDebug(debug);
		bnd.setDoNotCheckCapabilities(noCheck);
		
		return bnd;
	}
	
	public static Classifier createSMOBound() {
		SMOLinearBoundary smo = new SMOLinearBoundary();
		smo.setFilterType(new SelectedTag(SMO.FILTER_NONE, SMO.TAGS_FILTER));

		smo.setDebug(debug);
		smo.setDoNotCheckCapabilities(noCheck);
		
		return smo;
	}

	public static Classifier createLibSVMBound() {
		LibSVMSVCCLinearBoundary svm = new LibSVMSVCCLinearBoundary();
		svm.setCacheSize(100);

		svm.setDebug(debug);
		svm.setDoNotCheckCapabilities(noCheck);

		return svm;
	}

	public static Classifier createFLDABound(){
		FLDABoundary flda = new FLDABoundary();

		flda.setDebug(debug);
		flda.setDoNotCheckCapabilities(noCheck);

		return flda;
	}

	public static Classifier createMLPBoundary(){
		MultilayerPerceptronBoundary mlp = new MultilayerPerceptronBoundary();

		mlp.setDebug(debug);
		mlp.setDoNotCheckCapabilities(noCheck);

		return mlp;
	}

	public static Classifier createLogBound(){
		LogisticBoundary log = new LogisticBoundary();

		log.setDebug(debug);
		log.setDoNotCheckCapabilities(noCheck);

		return log;
	}


	
	public static Map<String,Classifier> createBaseClassifiers(){
	
		Map<String,Classifier> baseClassifierMap = new HashMap<>();
		
		baseClassifierMap.put("NC", createNCBound());
		
		baseClassifierMap.put("FLDA", createFLDABound());
		
		baseClassifierMap.put("LOG", createLogBound() );

		baseClassifierMap.put("SVM", createLibSVMBound() );
		
		
		return baseClassifierMap;
	}

	public static int nBaseClassifiers(){
		Map<String,Classifier> map = createBaseClassifiers();
		return map.size();
	}
	
	
	
	public static void writeJson2File(JSONArray jsonArray,String path2File) {
		FileWriter writer = null;
		
		try {
			File file = new File(path2File);
			file.getParentFile().mkdirs();
			file.createNewFile();
			
			writer = new FileWriter(path2File);
			jsonArray.write(writer,4,0);
		} catch (IOException e) {
			e.printStackTrace();
		}
		finally {
			
			try {
				if(writer !=null) {
					writer.flush();
					writer.close();
				}
				
			} catch (Exception e) {
				e.printStackTrace();
			}
			
		}
	}

	public static int nMethods(){
		Map<String,Classifier> bClassifiersMap = createBaseClassifiers();
		int nBase = bClassifiersMap.size();

		Map<String,Classifier> ensMap = createEnsemblesBagging(1);

		int nEnsembles = ensMap.size();

		int nMethods = nEnsembles/nBase;

		return nMethods;
	}

	public static int nMethodsBoosting(){
		Map<String,Classifier> bClassifiersMap = createBaseClassifiers();
		int nBase = bClassifiersMap.size();

		Map<String,Classifier> ensMap = createEnsemblesBoosting(1);

		int nEnsembles = ensMap.size();

		int nMethods = nEnsembles/nBase;

		return nMethods;
	}
	
	public static Map<String,Classifier> createEnsemblesBagging(int committeeSize){
		
		Map<String,Classifier> ensembleMap = new HashMap<>();
		
		Map<String,Classifier> baseClassifiers = createBaseClassifiers();
		for (Map.Entry<String, Classifier> entry : baseClassifiers.entrySet()) {
			
			String classifierName1 = "" + entry.getKey()+ "_MA";
			Classifier bagMA = createBaggingMA(entry.getValue(),committeeSize);
			
			ensembleMap.put(classifierName1, bagMA);

			String classifierName2 = "" + entry.getKey() + "_MV";
			Classifier bagMV = createBaggingMV(entry.getValue(),committeeSize);
			
			ensembleMap.put(classifierName2, bagMV);

			String classifierName3 = "" + entry.getKey() + "_TB";
			Classifier bagTB = createBaggingTanhBest(entry.getValue(),committeeSize);
			
			ensembleMap.put(classifierName3, bagTB);


			String classifierName4 = "" + entry.getKey() + "_BC";
			Classifier bagBC = createBaggingBoundAndCentroid(entry.getValue(),committeeSize);
			
			ensembleMap.put(classifierName4, bagBC);
			
			String classifierName5 = "" + entry.getKey() + "_BCs1";
			Classifier bagBCs1 = createBaggingBoundAndCentroids(entry.getValue(),committeeSize);
			
			ensembleMap.put(classifierName5, bagBCs1);

			String classifierName6 = "" + entry.getKey() + "_BCs2";
			Classifier bagBCs2 = createBaggingBoundAndCentroids2(entry.getValue(),committeeSize);
			
			ensembleMap.put(classifierName6, bagBCs2);

			String classifierName7 = "" + entry.getKey() + "_RT";
			Classifier bagRT= createBaggingRT(entry.getValue(),committeeSize);
			
			ensembleMap.put(classifierName7, bagRT);

			String classifierName8 = "" + entry.getKey() + "_RTP";
			Classifier bagRT_PCA= createBaggingRT_PCA(entry.getValue(),committeeSize);
			
			ensembleMap.put(classifierName8, bagRT_PCA);

		}
		
		return ensembleMap;
		
	}

	public static Map<String,Classifier> createEnsemblesBoosting(int committeeSize){
		
		Map<String,Classifier> ensembleMap = new HashMap<>();
		
		Map<String,Classifier> baseClassifiers = createBaseClassifiers();
		for (Map.Entry<String, Classifier> entry : baseClassifiers.entrySet()) {
			
			String classifierName1 = "" + entry.getKey()+ "_Si";
			Classifier boostingSign = createBoostingSign(entry.getValue(),committeeSize);
			ensembleMap.put(classifierName1, boostingSign);


			String classifierName2 = "" + entry.getKey()+ "_Ta";
			Classifier boostingTanh = createBoostingTanhBest(entry.getValue(),committeeSize);
			ensembleMap.put(classifierName2, boostingTanh);

			String classifierName3 = "" + entry.getKey()+ "_BC";
			Classifier boostingBndAndCentroid = createBoostingBoundAndCentroid(entry.getValue(),committeeSize);
			ensembleMap.put(classifierName3, boostingBndAndCentroid);

			String classifierName4 = "" + entry.getKey()+ "_BCs1";
			Classifier boostingBndAndCentroids1 = createBoostingBoundAndCentroids(entry.getValue(),committeeSize);
			ensembleMap.put(classifierName4, boostingBndAndCentroids1);

			String classifierName5 = "" + entry.getKey()+ "_BCs2";
			Classifier boostingBndAndCentroids2 = createBoostingBoundAndCentroids2(entry.getValue(),committeeSize);
			ensembleMap.put(classifierName5, boostingBndAndCentroids2);

			String classifierName6 = "" + entry.getKey()+ "_RT";
			Classifier boostingRT= createBoostingRT(entry.getValue(),committeeSize);
			ensembleMap.put(classifierName6, boostingRT);

			String classifierName7 = "" + entry.getKey()+ "_RTP";
			Classifier boostingRT_PCA= createBoostingRT_PCA(entry.getValue(),committeeSize);
			ensembleMap.put(classifierName7, boostingRT_PCA);
			
		}
		
		return ensembleMap;
		
	}
	 
	
		
	public static void createClassifiersConfigBagging(String outputPath,int committeeSize) {
		
		
		JSONArray jsonArray = new JSONArray();
		
		Map<String,Classifier> ensembles = createEnsemblesBagging(committeeSize);
		for( Map.Entry<String, Classifier> entry : ensembles.entrySet() ) {
			JSONObject jsonRep = new JSONObject(UtilsPT.getWekaSklearnMap(entry.getValue()));
			String classifierName = entry.getKey();
			jsonRep.put("name", classifierName);
			
			jsonArray.put(jsonRep);
		}
		
		writeJson2File(jsonArray, outputPath);

	}

	public static void createClassifiersConfigBoosting(String outputPath,int committeeSize) {
		
		
		JSONArray jsonArray = new JSONArray();
		
		Map<String,Classifier> ensembles = createEnsemblesBoosting(committeeSize);
		for( Map.Entry<String, Classifier> entry : ensembles.entrySet() ) {
			JSONObject jsonRep = new JSONObject(UtilsPT.getWekaSklearnMap(entry.getValue()));
			String classifierName = entry.getKey();
			jsonRep.put("name", classifierName);
			
			jsonArray.put(jsonRep);
		}
		
		writeJson2File(jsonArray, outputPath);

	}
	
	public static String createClassifiersConfigBagging2(int committeeSize) {
		
		
		JSONArray jsonArray = new JSONArray();
		
		Map<String,Classifier> ensembles = createEnsemblesBagging(committeeSize);
		for( Map.Entry<String, Classifier> entry : ensembles.entrySet() ) {
			JSONObject jsonRep = new JSONObject(UtilsPT.getWekaSklearnMap(entry.getValue()));
			String classifierName = entry.getKey();
			jsonRep.put("name", classifierName);
			
			jsonArray.put(jsonRep);
		}
		
		return jsonArray.toString();

	}


}
