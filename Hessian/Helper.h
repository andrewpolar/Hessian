#pragma once
#include <memory>
#include <vector>

typedef std::vector<std::vector<double>> Matrix;
typedef std::vector<double> Vector;

class Helper
{
public:
    static double Pearson(const std::unique_ptr<double[]>& x, const std::unique_ptr<double[]>& y, int len) {
        double xmean = 0.0;
        double ymean = 0.0;
        for (int i = 0; i < len; ++i) {
            xmean += x[i];
            ymean += y[i];
        }
        xmean /= len;
        ymean /= len;

        double covariance = 0.0;
        for (int i = 0; i < len; ++i) {
            covariance += (x[i] - xmean) * (y[i] - ymean);
        }

        double stdX = 0.0;
        double stdY = 0.0;
        for (int i = 0; i < len; ++i) {
            stdX += (x[i] - xmean) * (x[i] - xmean);
            stdY += (y[i] - ymean) * (y[i] - ymean);
        }
        stdX = sqrt(stdX);
        stdY = sqrt(stdY);
        return covariance / stdX / stdY;
    }
    static void ShowMatrix(std::unique_ptr<std::unique_ptr<double[]>[]>& matrix, int rows, int cols) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                printf("%5.3f ", matrix[i][j]);
            }
            printf("\n");
        }
    }
    static void ShowVector(std::unique_ptr<double[]>& ptr, int N) {
        int cnt = 0;
        for (int i = 0; i < N; ++i) {
            printf("%5.2f ", ptr[i]);
            if (++cnt >= 10) {
                printf("\n");
                cnt = 0;
            }
        }
    }
    static void SwapRows(std::unique_ptr<double[]>& row1, std::unique_ptr<double[]>& row2, int cols) {
        auto ptr = std::make_unique<double[]>(cols);
        for (int i = 0; i < cols; ++i) {
            ptr[i] = row1[i];
        }
        for (int i = 0; i < cols; ++i) {
            row1[i] = row2[i];
        }
        for (int i = 0; i < cols; ++i) {
            row2[i] = ptr[i];
        }
    }
    static void SwapScalars(double& x1, double& x2) {
        double buff = x1;
        x1 = x2;
        x2 = buff;
    }
    static void Shuffle(std::unique_ptr<std::unique_ptr<double[]>[]>& matrix, std::unique_ptr<double[]>& vector, int rows, int cols) {
        for (int i = 0; i < 2 * rows; ++i) {
            int n1 = rand() % rows;
            int n2 = rand() % rows;
            SwapRows(matrix[n1], matrix[n2], cols);
            SwapScalars(vector[n1], vector[n2]);
        }
    }
    static void FindMinMaxMatrix(std::vector<double>& xmin, std::vector<double>& xmax,
        std::unique_ptr<std::unique_ptr<double[]>[]>& matrix, int nRows, int nCols) {
        for (int i = 0; i < nCols; ++i) {
            xmin.push_back(DBL_MAX);
            xmax.push_back(-DBL_MAX);
        }
        for (int i = 0; i < nRows; ++i) {
            for (int j = 0; j < nCols; ++j) {
                if (matrix[i][j] < xmin[j]) xmin[j] = static_cast<double>(matrix[i][j]);
                if (matrix[i][j] > xmax[j]) xmax[j] = static_cast<double>(matrix[i][j]);
            }
        }
    }

    static void FindMinMax(std::vector<double>& xmin, std::vector<double>& xmax,
        double& targetMin, double& targetMax,
        std::unique_ptr<std::unique_ptr<double[]>[]>& matrix,
        std::unique_ptr<double[]>& target, int nRows, int nCols) {

        for (int i = 0; i < nCols; ++i) {
            xmin.push_back(DBL_MAX);
            xmax.push_back(-DBL_MAX);
        }
        for (int i = 0; i < nRows; ++i) {
            for (int j = 0; j < nCols; ++j) {
                if (matrix[i][j] < xmin[j]) xmin[j] = static_cast<double>(matrix[i][j]);
                if (matrix[i][j] > xmax[j]) xmax[j] = static_cast<double>(matrix[i][j]);
            }
        }
        targetMin = DBL_MAX;
        targetMax = -DBL_MAX;
        for (int j = 0; j < nRows; ++j) {
            if (target[j] < targetMin) targetMin = target[j];
            if (target[j] > targetMax) targetMax = target[j];
        }
    }

    static double Min(const std::unique_ptr<double[]>& x, int N) {
        double min = x[0];
        for (int i = 1; i < N; ++i) {
            if (x[i] < min) min = x[i];
        }
        return min;
    }

    static double Max(const std::unique_ptr<double[]>& x, int N) {
        double max = x[0];
        for (int i = 1; i < N; ++i) {
            if (x[i] > max) max = x[i];
        }
        return max;
    }

    static double MinV(const std::vector<double>& x) {
        double min = x[0];
        for (int i = 1; i < (int)x.size(); ++i) {
            if (x[i] < min) min = x[i];
        }
        return min;
    }

    static double MaxV(const std::vector<double>& x) {
        double max = x[0];
        for (int i = 1; i < (int)x.size(); ++i) {
            if (x[i] > max) max = x[i];
        }
        return max;
    }

    static Matrix MakeMTM(const std::unique_ptr<std::unique_ptr<double[]>[]>& M, int nRows, int nCols) {
        Matrix MTM = Matrix(nCols, Vector(nCols, 0.0));
        for (int i = 0; i < nCols; ++i) {
            for (int j = 0; j < nCols; ++j) {
                MTM[i][j] = 0.0;
                for (int k = 0; k < nRows; ++k) {
                    MTM[i][j] += M[k][i] * M[k][j];
                }
            }
        }
        double trace = 0;
        for (int i = 0; i < nCols; ++i) {
            trace += MTM[i][i];
        }
        trace /= nCols;
        trace /= 1000.0;
        for (int i = 0; i < nCols; ++i) {
            MTM[i][i] += trace;
        }
        return MTM;
    }

    static Vector MakeMTY(const std::unique_ptr<std::unique_ptr<double[]>[]>& M, const std::unique_ptr<double[]>& Y, int nRows, int nCols) {
        Vector MTY = Vector(nCols, 0.0);
        for (int i = 0; i < nCols; ++i) {
            for (int k = 0; k < nRows; ++k) {
                MTY[i] += M[k][i] * Y[k];
            }
        }
        return MTY;
    }

    static void ShowMatrix(Matrix M) {
        for (int i = 0; i < (int)M.size(); ++i) {
            for (int j = 0; j < (int)M[i].size(); ++j) {
                printf("%5.3f ", M[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    static void ShowVector(Vector V) {
        int cnt = 0;
        for (int i = 0; i < (int)V.size(); ++i) {
            printf("%5.2f ", V[i]);
            if (++cnt >= 10) {
                printf("\n");
                cnt = 0;
            }
        }
        printf("\n");
    }
    static void AddRegularization(Matrix& M, double denominator) {
        double trace = 0;
        for (int i = 0; i < (int)M.size(); ++i) {
            trace += M[i][i];
        }
        trace /= M.size();
        trace /= denominator;
        for (int i = 0; i < (int)M.size(); ++i) {
            M[i][i] += trace;
        }
    }
    static Vector tAmAs(Matrix& A, Vector& y) {
        int size = (int)y.size();
        double trace = 0.0;
        for (int i = 0; i < size; ++i) {
            trace += A[i][i];
        }
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                A[i][j] /= trace;
            }
        }
        for (int i = 0; i < size; ++i) {
            y[i] /= trace;
        }
        Vector x = Vector(size, 0.0);
        for (int i = 0; i < size; ++i) {
            x[i] = 2.0 * y[i];
        }
        for (int i = 0; i < size; ++i) {
            for (int j = 0; j < size; ++j) {
                x[i] -= A[i][j] * y[j];
            }
        }
        return x;
    }
};
