//Concept: Andrew Polar and Mike Poluektov
//Developer Andrew Polar

//License
//In case if end user finds the way of making a profit by using this code and earns
//billions of US dollars and meet developer bagging change in the street near McDonalds,
//he or she is not in obligation to buy him a sandwich.

//Symmetricity
//In case developer became rich and famous by publishing this code and meet misfortunate
//end user who went bankrupt by using this code, he is also not in obligation to buy
//end user a sandwich.

//Publications:
//https://www.sciencedirect.com/science/article/abs/pii/S0016003220301149
//https://www.sciencedirect.com/science/article/abs/pii/S0952197620303742
//https://arxiv.org/abs/2305.08194

//Remarks from Andrew Polar.
//This is comparison of two methods Newton-Kaczmarz and LBFGS.
//The executable Hessian.exe uses library provided by Jorge Nocedal and Naoaki Okazaki.
//I use library as is whithout changes. Library must be compiled prior to executable. 
//I know how to set the order of building projects, but it looks like Microsoft developer don't 
//know how to preserve this order in solution file, so you have to maintain this order 
//manually. Right-click and build library first, then right-click and build executable, 
//don't use rebuild command. Sorry for inconvenience, please redirect your anger to Microsoft. 

#include <iostream>
#include <stdio.h>
#include "lbfgs.h"
#include "Helper.h"
#include "Urysohn.h"

std::unique_ptr<std::unique_ptr<double[]>[]> MakeRandomMatrix(int rows, int cols, double min, double max) {
	auto matrix = std::make_unique<std::unique_ptr<double[]>[]>(rows);
	for (int i = 0; i < rows; ++i) {
		matrix[i] = std::make_unique<double[]>(cols);
		for (int j = 0; j < cols; ++j) {
			matrix[i][j] = static_cast<double>((rand() % 1000) / 1000.0) * (max - min) + min;
		}
	}
	return matrix;
}

std::unique_ptr<double[]> ComputeTarget(const std::unique_ptr<std::unique_ptr<double[]>[]>& X, int rows, int nCols) {
	auto vector = std::make_unique<double[]>(rows);
	for (int i = 0; i < rows; ++i) {
		double s = 0.0;
		for (int j = 0; j < nCols; ++j) {
			s += X[i][j] * X[i][j];
		}
		vector[i] = sqrt(s);
	}
	return vector;
}

double GetGradientAndObjectiveFunction(const std::unique_ptr<Urysohn>& u, const std::unique_ptr<std::unique_ptr<double[]>[]>& features,
	const std::unique_ptr<double[]>& targets, int nRecords, double* g) {

	double fx = 0.0;
	int size = (int)u->_model.size() * (int)u->_model[0].size();
	for (int j = 0; j < size; ++j) g[j] = 0.0;
	for (int i = 0; i < nRecords; ++i) {
		double m = u->GetJacobianElementsAndUrysohn(features[i]);
		double residual = targets[i] - m;
		fx += residual * residual;
		for (int j = 0; j < (int)u->_indexes.size(); ++j) {
			int k = u->_indexes[j];
			g[k] += -2.0 * (residual)*u->_derivatives[j];
		}
	}
	return fx;
}

double AccuracyAssessment(const std::unique_ptr<Urysohn>& u, const std::unique_ptr<std::unique_ptr<double[]>[]>& features,
	const std::unique_ptr<double[]>& targets, int nRecords) {

	double error = 0.0;
	for (int i = 0; i < nRecords; ++i) {
		double m = u->GetUrysohn(features[i]);
		double residual = targets[i] - m;
		error += residual * residual;
	}
	error /= nRecords;
	error = sqrt(error);
	return error;
}

//this class is needed only to pass objects to callbacks of LBFGS lib
class ObjectsHolder {
public:
	int _nTrainingRecords;
	int _nValidationRecords;
	std::unique_ptr<std::unique_ptr<double[]>[]> _features_training;
	std::unique_ptr<std::unique_ptr<double[]>[]> _features_validation;
	std::unique_ptr<double[]> _targets_training;
	std::unique_ptr<double[]> _targets_validation;
	std::unique_ptr<Urysohn> _u;
	double _targetMin;
	double _targetMax;
	ObjectsHolder(int nTrainingRecords, int nValidationRecords, int nFeatures, int nPoints, double min, double max) {
		_nTrainingRecords = nTrainingRecords;
		_nValidationRecords = nValidationRecords;
		_nFeatures = nFeatures;
		_nPoints = nPoints;
		_min = min;
		_max = max;
		_features_training = MakeRandomMatrix(_nTrainingRecords, _nFeatures, _min, _max);
		_features_validation = MakeRandomMatrix(_nValidationRecords, _nFeatures, _min, _max);
		_targets_training = ComputeTarget(_features_training, _nTrainingRecords, _nFeatures);
		_targets_validation = ComputeTarget(_features_validation, _nValidationRecords, _nFeatures);

		std::vector<double> argmin;
		std::vector<double> argmax;
		Helper::FindMinMax(argmin, argmax, _targetMin, _targetMax, _features_training, _targets_training,
			_nTrainingRecords, _nFeatures);

		_u = std::make_unique<Urysohn>(argmin, argmax, _targetMin, _targetMax, _nPoints);
	}
private:
	int _nFeatures;
	int _nPoints;
	double _min = 0.0;
	double _max = 10.0;
};

lbfgsfloatval_t evaluate(void* instance, const lbfgsfloatval_t* x, lbfgsfloatval_t* g, const int n, const lbfgsfloatval_t step) {
	ObjectsHolder* ctx = (ObjectsHolder*)instance;

	ctx->_u->AssignUrysohnDirect(x, n);
	double fx = GetGradientAndObjectiveFunction(ctx->_u, ctx->_features_training, ctx->_targets_training, ctx->_nTrainingRecords, g);

	return fx;
}

int progress(void* instance,
	const lbfgsfloatval_t* x, const lbfgsfloatval_t* g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
	const lbfgsfloatval_t step, int n, int k, int ls) {
	
	ObjectsHolder* ctx = (ObjectsHolder*)instance;
	double error2 = AccuracyAssessment(ctx->_u, ctx->_features_validation, ctx->_targets_validation, ctx->_nValidationRecords);
	double error = error2 / (ctx->_targetMax - ctx->_targetMin);
	printf("RRMSE for validation = %f\n", error);

	if (error < 0.011) return 1;
	return 0;
}

void RunLBFGS() {
	clock_t start_application = clock();
	clock_t current_time = clock();

	int nFeatures = 24;
	int nPoints = 8;
	int N = nFeatures * nPoints;  
	lbfgsfloatval_t fx;
	lbfgsfloatval_t* x = lbfgs_malloc(N);
	lbfgs_parameter_t param;

	if (x == NULL) {
		printf("ERROR: Failed to allocate a memory block for variables.\n");
		exit(1);
	}

	for (int i = 0; i < N; ++i) {
		x[i] = rand() % 1000 / 1000.0;
	}

	lbfgs_parameter_init(&param);
	printf("LBGFS\n");
	ObjectsHolder* objectsholder = new ObjectsHolder(2000, 200, nFeatures, nPoints, 0.0, 10.0);
	int ret = lbfgs(N, x, &fx, evaluate, progress, objectsholder, &param);

	lbfgs_free(x);
	delete objectsholder;

	current_time = clock();
	printf("Training time for LBFGS %f\n\n", (double)(current_time - start_application) / CLOCKS_PER_SEC);
}

void RunKaczmarz() {
	const int nTrainingRecords = 2000;
	const int nValidationRecords = 200;
	const int nFeatures = 24;
	const double min = 0.0;
	const double max = 10.0;
	const double mu = 0.01;
	const int nPoints = 8;
	const int nEpochs = 64;

	//generating data
	auto features_training = MakeRandomMatrix(nTrainingRecords, nFeatures, min, max);
	auto features_validation = MakeRandomMatrix(nValidationRecords, nFeatures, min, max);
	auto targets_training = ComputeTarget(features_training, nTrainingRecords, nFeatures);
	auto targets_validation = ComputeTarget(features_validation, nValidationRecords, nFeatures);

	clock_t start_application = clock();
	clock_t current_time = clock();

	std::vector<double> argmin;
	std::vector<double> argmax;
	double targetMin;
	double targetMax;
	Helper::FindMinMax(argmin, argmax, targetMin, targetMax, features_training, targets_training,
		nTrainingRecords, nFeatures);

	auto u = std::make_unique<Urysohn>(argmin, argmax, targetMin, targetMax, nPoints);

	printf("Kaczmarz training\n");
	for (int epoch = 0; epoch < nEpochs; ++epoch) {
		for (int i = 0; i < nTrainingRecords; ++i) {
			double m = u->GetUrysohn(features_training[i]);
			double residual = targets_training[i] - m;
			u->Update(residual * mu, features_training[i]);
		}

		double error3 = AccuracyAssessment(u, features_validation, targets_validation, nValidationRecords);
		current_time = clock();
		printf("RRMSE for validation = %f, epoch = %d, training time %f\n", error3 / (targetMax - targetMin), epoch,
			(double)(current_time - start_application) / CLOCKS_PER_SEC);

		if (error3 / (targetMax - targetMin) < 0.011) break;
	}
}

int main() {
	srand((unsigned int)time(NULL));
	RunLBFGS();
	RunKaczmarz();
	return 0;
}

