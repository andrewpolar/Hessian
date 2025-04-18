#pragma once
#include <memory>
#include <vector>

class Urysohn {
public:
	Urysohn(const std::vector<double>& argmin, const std::vector<double>& argmax, double umin, double umax, int nPoints) {
		if (argmin.size() != argmax.size()) {
			printf("Fatal: argument sizes mismatch");
			exit(0);
		}
		int nFunctions = (int)argmin.size();
		double fmin = umin / nFunctions;
		double fmax = umax / nFunctions;
		_model = std::vector<std::vector<double>>(nFunctions);
		for (int i = 0; i < nFunctions; ++i) {
			_model[i] = std::vector<double>(nPoints);
			for (int j = 0; j < nPoints; ++j) {
				_model[i][j] = (rand() % 1000 / 1000.0) * (fmax - fmin) + fmin;
			}
		}
		_xmin = std::vector<double>(nFunctions);
		_xmax = std::vector<double>(nFunctions);
		_deltax = std::vector<double>(nFunctions);
		for (int i = 0; i < nFunctions; ++i) {
			_xmin[i] = argmin[i];
			_xmax[i] = argmax[i];
			SetLimits(i);
		}
		_leftIndex = std::vector<int>(nFunctions);
		_rightIndex = std::vector<int>(nFunctions);
		_leftOffset = std::vector<double>(nFunctions);
		_rightOffset = std::vector<double>(nFunctions);

		_indexes.resize(nFunctions * 2);
		_derivatives.resize(nFunctions * 2);
		_indexes.assign(_indexes.size(), 0);
		_derivatives.assign(_derivatives.size(), 0.0);
	}
	Urysohn(const Urysohn& uri) {
		_xmin.clear();
		_xmin = std::vector<double>(uri._xmin.size());
		for (int i = 0; i < (int)uri._xmin.size(); ++i) {
			_xmin[i] = uri._xmin[i];
		}
		_xmax.clear();
		_xmax = std::vector<double>(uri._xmax.size());
		for (int i = 0; i < (int)uri._xmax.size(); ++i) {
			_xmax[i] = uri._xmax[i];
		}
		_deltax.clear();
		_deltax = std::vector<double>(uri._deltax.size());
		for (int i = 0; i < (int)uri._deltax.size(); ++i) {
			_deltax[i] = uri._deltax[i];
		}
		_model.clear();
		_model = std::vector<std::vector<double>>(uri._model.size());
		for (int i = 0; i < (int)uri._model.size(); ++i) {
			_model[i] = std::vector<double>(uri._model[i].size());
			for (int j = 0; j < (int)uri._model[i].size(); ++j) {
				_model[i][j] = uri._model[i][j];
			}
		}
	}
	double GetJacobianElementsAndUrysohn(const std::unique_ptr<double[]>& inputs) {
		double f = 0.0;
		for (int i = 0; i < (int)_model.size(); ++i) {
			f += GetJacobianElementsAndFunction(i, inputs[i]);
		}
		UpdateJacobian();
		return f;
	}
	double GetUrysohn(const std::unique_ptr<double[]>& inputs, std::unique_ptr<double[]>& derivatives) {
		double f = 0.0;
		for (int i = 0; i < (int)_model.size(); ++i) {
			f += GetFunction(i, inputs[i], derivatives[i]);
		}
		return f;
	}
	double GetUrysohn(const std::unique_ptr<double[]>& inputs) {
		double f = 0.0;
		for (int i = 0; i < (int)_model.size(); ++i) {
			f += GetFunction(i, inputs[i]);
		}
		return f;
	}
	void Update(double delta, const std::unique_ptr<double[]>& inputs) {
		for (int i = 0; i < (int)_model.size(); ++i) {
			Update(i, inputs[i], delta);
		}
	}
	void IncrementPoints() {
		for (int i = 0; i < (int)_model.size(); ++i) {
			IncrementPoints(i);
		}
	}
	void ShowData() {
		printf("Min, max, delta\n");
		for (int i = 0; i < (int)_xmin.size(); ++i) {
			printf("%7.4f ", _xmin[i]);
		}
		printf("\n");
		for (int i = 0; i < (int)_xmax.size(); ++i) {
			printf("%7.4f ", _xmax[i]);
		}
		printf("\n");
		for (int i = 0; i < (int)_deltax.size(); ++i) {
			printf("%7.4f ", _deltax[i]);
		}
		printf("\n\n");
		printf("Urysohn: rows = functions, cols = points\n");
		for (int i = 0; i < (int)_model.size(); ++i) {
			for (int j = 0; j < (int)_model[i].size(); ++j) {
				printf("%7.4f ", _model[i][j]);
			}
			printf("\n");
		}
		printf("\n");
	}

	void UpdateUrysohnDirect(std::vector<double> x) {
		int nFeatures = (int)_model.size();
		int n = (int)x.size() / nFeatures;
		int k = 0;
		int nWhich = 0;
		for (int i = 0; i < (int)x.size(); ++i) {
			_model[nWhich][k] += x[i];
			++k;
			if (k >= n) {
				nWhich += 1;
				k = 0;
			}
		}
	}

	void AssignUrysohnDirect(const double* x, int N) {
		int nFeatures = (int)_model.size();
		int n = N / nFeatures;
		int k = 0;
		int nWhich = 0;
		for (int i = 0; i < N; ++i) {
			_model[nWhich][k] = x[i];
			++k;
			if (k >= n) {
				nWhich += 1;
				k = 0;
			}
		}
	}

	std::vector<int> _indexes;
	std::vector<double> _derivatives;
	std::vector<std::vector<double>> _model;
private:
	std::vector<double> _xmin;
	std::vector<double> _xmax;
	std::vector<double> _deltax;
	//
	std::vector<int> _leftIndex;
	std::vector<int> _rightIndex;
	std::vector<double> _leftOffset;
	std::vector<double> _rightOffset;
	//
	void SetLimits(int k) {
		double range = _xmax[k] - _xmin[k];
		_xmin[k] -= 0.01 * range;
		_xmax[k] += 0.01 * range;
		_deltax[k] = (_xmax[k] - _xmin[k]) / (_model[k].size() - 1);
	}
	void IncrementPoints(int k) {
		int points = (int)_model[k].size() + 1;
		double deltax = (_xmax[k] - _xmin[k]) / (points - 1);
		std::vector<double> y(points);
		y[0] = _model[k][0];
		y[points - 1] = _model[k][_model[k].size() - 1];
		for (int i = 1; i < points - 1; ++i) {
			y[i] = GetFunction(k, _xmin[k] + i * deltax);
		}
		_deltax[k] = deltax;
		_model[k].clear();
		for (int i = 0; i < (int)y.size(); i++)
		{
			_model[k].push_back(y[i]);
		}
	}
	void Update(int k, double x, double residual) {
		if (x < _xmin[k]) {
			_xmin[k] = x;
			SetLimits(k);
		}
		if (x > _xmax[k]) {
			_xmax[k] = x;
			SetLimits(k);
		}
		double R = (x - _xmin[k]) / _deltax[k];
		int index = (int)(R);
		double offset = R - index;
		double tmp = residual * offset;
		_model[k][index + 1] += tmp;
		_model[k][index] += residual - tmp;
	}
	double GetFunction(int k, double x, double& derivative) {
		if (x <= _xmin[k]) {
			int index = 0;
			derivative = (_model[k][index + 1] - _model[k][index]) / _deltax[k];
			return _model[k][0];
		}
		if (x >= _xmax[k]) {
			int index = (int)_model[k].size() - 2;
			derivative = (_model[k][index + 1] - _model[k][index]) / _deltax[k];
			return _model[k][_model[k].size() - 1];
		}
		double R = (x - _xmin[k]) / _deltax[k];
		int index = (int)(R);
		derivative = (_model[k][index + 1] - _model[k][index]) / _deltax[k];
		double offset = R - index;
		return _model[k][index] + (_model[k][index + 1] - _model[k][index]) * offset;
	}
	double GetFunction(int k, double x) {
		if (x <= _xmin[k]) {
			return _model[k][0];
		}
		if (x >= _xmax[k]) {
			return _model[k][_model[k].size() - 1];
		}
		double R = (x - _xmin[k]) / _deltax[k];
		int index = (int)(R);
		double offset = R - index;
		return _model[k][index] + (_model[k][index + 1] - _model[k][index]) * offset;
	}
	double GetJacobianElementsAndFunction(int k, double x) {
		if (x <= _xmin[k]) {
			int index = 0;
			_leftIndex[k] = index;
			_rightIndex[k] = index + 1;
			_leftOffset[k] = 0.0;
			_rightOffset[k] = 1.0;
			return _model[k][0];
		}
		if (x >= _xmax[k]) {
			int index = (int)_model[k].size() - 2;
			_leftIndex[k] = index;
			_rightIndex[k] = index + 1;
			_leftOffset[k] = 1.0;
			_rightOffset[k] = 0.0;
			return _model[k][_model[k].size() - 1];
		}
		double R = (x - _xmin[k]) / _deltax[k];
		int index = (int)(R);
		double offset = R - index;
		_leftIndex[k] = index;
		_rightIndex[k] = index + 1;
		_leftOffset[k] = offset;
		_rightOffset[k] = 1.0 - offset;
		return _model[k][index] + (_model[k][index + 1] - _model[k][index]) * offset;
	}
	void UpdateJacobian() {
		int nFeatures = (int)_model.size();
		int nPoints = (int)_model[0].size();
		int counterIndexes = 0;
		int counterDerivatives = 0;
		for (int i = 0; i < nFeatures; ++i) {
			_indexes[counterIndexes++] = _leftIndex[i] + nPoints * i;
			_indexes[counterIndexes++] = _rightIndex[i] + nPoints * i;
			_derivatives[counterDerivatives++] = _rightOffset[i];
			_derivatives[counterDerivatives++] = _leftOffset[i];
		}
	}
};

