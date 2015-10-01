//
//  Zinnia: Online hand recognition system with machine learning
//
//  $Id$;
//
//  Copyright(C) 2008 Taku Kudo <taku@chasen.org>
//
#include <cstring>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include "feature.h"

namespace zinnia {

namespace {
static const double kEPS = 0.1;
static const double kINF = 1e+37;
}

bool svm_train(size_t l,
               size_t n,
               const float *y,
               const FeatureNode **x,
               double C,
               double *w) {
  //l: y.size number of characters to train
  //n: w.size
  //*y: vector of +1.0 and -1.0
  //**x: vector of features
  //C: 1.0 <- regularization parameter 
  //*w: empty vector of length n
  size_t active_size = l;
  double PGmax_old = kINF;
  double PGmin_old = -kINF;
  std::vector<double> QD(l);
  std::vector<size_t> index(l);
  std::vector<double> alpha(l);

  //length n is max_dim + 1, set in Trainer::add(character)
  std::fill(w, w + n, 0.0);
  std::fill(alpha.begin(), alpha.end(), 0.0);

  //Newton-Raphson method is used below to determine alpha with the inequality constraint:
  // 0 <= alpha <= C

  //QD = x'x
  //dotproduct of feature vector with itself which serves probably as a constant representing the 2. partial derivative of the Lagrangian Term?
  for (size_t i = 0; i < l; ++i) {
    index[i] = i;
    QD[i] = 0;
    for (const FeatureNode *f = x[i]; f->index >= 0; ++f) {
      QD[i] += (f->value * f->value);
    }
  }

  static const size_t kMaxIteration = 2000;
  for (size_t iter = 0; iter < kMaxIteration; ++iter) {
    double PGmax_new = -kINF;
    double PGmin_new = kINF;
    //random shuffle index from 0 to l
    //rearrange the elements in the index
    //what is the purpose of this?
    std::random_shuffle(index.begin(), index.begin() + active_size);

    //for each character represented by i
    //calculate the coefficient alpha according to the Newton-Raphson method:
    //alpha_new = alpha_old - L(alpha)'/L(alpha)''
    //and the corresponding w = sum(alpha_i y_i x_i)
    for (size_t s = 0; s < active_size; ++s) {
      const size_t i = index[s];
      double G = 0;

      std::cout << "Index" << i << "\n";
      for (const FeatureNode *f = x[i]; f->index >= 0; ++f) {
        G += w[f->index] * f->value;
      }

      G = G * y[i] - 1;
      double PG = 0.0;

      if (alpha[i] == 0.0) {
        if (G > PGmax_old) {
          active_size--;
          std::swap(index[s], index[active_size]);
          s--;
          continue;
        } else if (G < 0.0) {
          PG = G;
        }
      } else if (alpha[i] == C) {
        if (G < PGmin_old) {
          active_size--;
          std::swap(index[s], index[active_size]);
          s--;
          continue;
        } else if (G > 0.0) {
          PG = G;
        }
      } else {
        PG = G;
      } 

      PGmax_new = std::max(PGmax_new, PG);
      PGmin_new = std::min(PGmin_new, PG);

      if (std::abs(PG) > 1.0e-12) {
        const double alpha_old = alpha[i];
        //the constraint 0 <= alpha <= C
        //alpha_new = alpha_old - G/QD
        alpha[i] = std::min(std::max(alpha[i] - G/QD[i], 0.0), C);

        //d = G/QD * y[i]
        //why subtract alpha_old?
        const double d = (alpha[i] - alpha_old)* y[i];
        for (const FeatureNode *f = x[i]; f->index >= 0; ++f) {
          w[f->index] += d * f->value;
        }
      }
      
    }

    if (iter % 4 == 0) {
      std::cout << "." << std::flush;
    }

    if ((PGmax_new - PGmin_new) <= kEPS) {
      if (active_size == l) {
        break;
      } else {
        active_size = l;
        PGmax_old = kINF;
        PGmin_old = -kINF;
        continue;
      }
    }

    PGmax_old = PGmax_new;
    PGmin_old = PGmin_new;
    if (PGmax_old <= 0) {
      PGmax_old = kINF;
    }
    if (PGmin_old >= 0) {
      PGmin_old = -kINF;
    }
  }

  std::cout << std::endl;

  return true;
}
}
