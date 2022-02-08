//Code borrowed and adapted from tiny_dnn
#pragma once

#include <algorithm>
#include <unordered_map>

#undef min
#undef max

namespace ToyDNN
{

    struct Optimizer
    {
        virtual ~Optimizer() = default;
        virtual void UpdateTrainableParameters( const std::vector<Scalar>& dW, std::vector<Scalar>& W ) = 0;
        virtual void reset() {}  // override to implement pre-learning action
    };

    template <int N>
    struct StatefulOptimizer : public Optimizer
    {
        void reset() override
        {
            for( auto& e : E_ ) e.clear();
        }

    protected:
        template <int Index>
        std::vector<Scalar>& get( const std::vector<Scalar>& key )
        {
            static_assert(Index < N, "index out of range");
            if( E_[Index][&key].empty() ) E_[Index][&key].resize( key.size(), Scalar() );
            return E_[Index][&key];
        }
        std::unordered_map<const std::vector<Scalar>*, std::vector<Scalar>> E_[N];
    };

    /**
     * adaptive gradient method
     *
     * J Duchi, E Hazan and Y Singer,
     * Adaptive subgradient methods for online learning and stochastic optimization
     * The Journal of Machine Learning Research, pages 2121-2159, 2011.
     **/
    struct AdagradOptimizer : public StatefulOptimizer<1>
    {
        AdagradOptimizer() : LearningRate( Scalar( 0.01 ) ), eps( Scalar( 1e-8 ) ) {}

        void UpdateTrainableParameters( const std::vector<Scalar>& dW, std::vector<Scalar>& W )
        {
            std::vector<Scalar>& g = get<0>( W );
            for( uint32_t i = 0 ; i < W.size() ; ++i )
            {
                g[i] += dW[i] * dW[i];
                W[i] -= LearningRate * dW[i] / (std::sqrt( g[i] ) + eps);
            }
        }

        Scalar LearningRate;  // learning rate
    private:
        Scalar eps;
    };

    /**
     * RMSprop
     *
     * T Tieleman, and G E Hinton,
     * Lecture 6.5 - rmsprop, COURSERA: Neural Networks for Machine Learning (2012)
     **/
    struct RMSpropOptimizer : public StatefulOptimizer<1>
    {
        RMSpropOptimizer() : LearningRate( Scalar( 0.0001 ) ), mu( Scalar( 0.99 ) ), eps( Scalar( 1e-8 ) ) {}

        void UpdateTrainableParameters( const std::vector<Scalar>& dW, std::vector<Scalar>& W )
        {
            std::vector<Scalar>& g = get<0>( W );

            for( uint32_t i = 0 ; i < W.size() ; ++i )
            {
                g[i] = mu * g[i] + (1 - mu) * dW[i] * dW[i];
                W[i] -= LearningRate * dW[i] / std::sqrt( g[i] + eps );
            }
        }

        Scalar LearningRate;
        Scalar mu;     // decay term
    private:
        Scalar eps;  // constant value to avoid zero-division
    };

    /**
     * @brief [a new optimizer (2015)]
     * @details [see Adam: A Method for Stochastic Optimization (Algorithm 1)
     *               http://arxiv.org/abs/1412.6980]
     *
     */
    struct AdamOptimizer : public StatefulOptimizer<2>
    {
        AdamOptimizer()
            : LearningRate( Scalar( 0.001 ) ),
            b1( Scalar( 0.9 ) ),
            b2( Scalar( 0.999 ) ),
            b1_t( Scalar( 0.9 ) ),
            b2_t( Scalar( 0.999 ) ),
            eps( Scalar( 1e-8 ) )
        {
        }

        void UpdateTrainableParameters( const std::vector<Scalar>& dW, std::vector<Scalar>& W )
        {
            std::vector<Scalar>& mt = get<0>( W );
            std::vector<Scalar>& vt = get<1>( W );

            for( uint32_t i = 0 ; i < W.size() ; ++i )
            {
                mt[i] = b1 * mt[i] + (Scalar( 1 ) - b1) * dW[i];
                vt[i] = b2 * vt[i] + (Scalar( 1 ) - b2) * dW[i] * dW[i];

                // L2 norm based update rule
                W[i] -= LearningRate * (mt[i] / (Scalar( 1 ) - b1_t)) /
                    std::sqrt( (vt[i] / (Scalar( 1 ) - b2_t)) + eps );
            }

            b1_t *= b1;
            b2_t *= b2;
        }

        Scalar LearningRate;
        Scalar b1;     // decay term
        Scalar b2;     // decay term
        Scalar b1_t;   // decay term power t
        Scalar b2_t;   // decay term power t

    private:
        Scalar eps;  // constant value to avoid zero-division
    };

    /**
     * @brief [a new optimizer (2015)]
     * @details [see Adam: A Method for Stochastic Optimization (Algorithm 2)
     *               http://arxiv.org/abs/1412.6980]
     *
     */
    struct AdamaxOptimizer : public StatefulOptimizer<2>
    {
        AdamaxOptimizer()
            : 
            LearningRate( Scalar( 0.002 ) ),
            b1( Scalar( 0.9 ) ),
            b2( Scalar( 0.999 ) ),
            b1_t( b1 ),
            eps( Scalar( 1e-8 ) )
        {
        }

        void UpdateTrainableParameters( const std::vector<Scalar>& dW, std::vector<Scalar>& W )
        {
            std::vector<Scalar>& mt = get<0>( W );
            std::vector<Scalar>& ut = get<1>( W );

            for( uint32_t i = 0 ; i < W.size() ; ++i )
            {
                mt[i] = b1 * mt[i] + (Scalar( 1 ) - b1) * dW[i];
                ut[i] = std::max( b2 * ut[i], std::abs( dW[i] ) );

                // Lp norm based update rule
                W[i] -= (LearningRate / (Scalar(1.0) - b1_t)) * (mt[i] / (ut[i] + eps));
            }

            b1_t *= b1;
        }

        Scalar LearningRate;  // learning rate
        Scalar b1;     // decay term
        Scalar b2;     // decay term
        Scalar b1_t;   // decay term power t

    private:
        Scalar eps;  // constant value to avoid zero-division
    };

    /**
     * SGD without momentum
     *
     **/
    struct SGDOptimizer : public Optimizer
    {
        SGDOptimizer() : LearningRate( Scalar( 0.01 ) ), WeightDecay( Scalar( 0 ) ) {}

        void UpdateTrainableParameters( const std::vector<Scalar>& dW, std::vector<Scalar>& W )
        {
            for( uint32_t i = 0 ; i < W.size() ; ++i )
            {
                W[i] = W[i] - LearningRate * (dW[i] + WeightDecay * W[i]);
            }
        }

        Scalar LearningRate;
        Scalar WeightDecay;  // Similar to L2 regularization
    };

    /**
     * SGD with momentum
     *
     * B T Polyak,
     * Some methods of speeding up the convergence of iteration methods
     * USSR Computational Mathematics and Mathematical Physics, 4(5):1-17, 1964.
     **/
    struct MomentumSGDOptimizer : public StatefulOptimizer<1>
    {
    public:
        MomentumSGDOptimizer() : LearningRate( Scalar( 0.01 ) ), WeightDecay( Scalar( 0 ) ), Momentum( Scalar( 0.9 ) ) {}

        void UpdateTrainableParameters( const std::vector<Scalar>& dW, std::vector<Scalar>& W )
        {
            std::vector<Scalar>& dWprev = get<0>( W );

            for( uint32_t i = 0 ; i < W.size() ; ++i )
            {
                Scalar V = Momentum * dWprev[i] - LearningRate * (dW[i] + W[i] * WeightDecay);
                W[i] += V;
                dWprev[i] = V;
            }
        }

        Scalar LearningRate;
        Scalar WeightDecay;
        Scalar Momentum;
    };

    /**
     * SGD with Nesterov momentum
     *
     * Y Nesterov,
     * A method for unconstrained convex minimization problem with the rate of
     * convergence o(1/k2), Doklady ANSSSR, vol.269, pp.543-547, 1983.
     **/
    struct NesterovMomentumOptimizer : public StatefulOptimizer<1>
    {
    public:
        NesterovMomentumOptimizer()
            : LearningRate( Scalar( 0.01 ) ), WeightDecay( Scalar( 0 ) ), Momentum( Scalar( 0.9 ) )
        {
        }

        void UpdateTrainableParameters( const std::vector<Scalar>& dW, std::vector<Scalar>& W )
        {
            std::vector<Scalar>& dWprev = get<0>( W );

            for( uint32_t i = 0 ; i < W.size() ; ++i )
            {
                Scalar V = Momentum * dWprev[i] - LearningRate * (dW[i] + W[i] * WeightDecay);
                W[i] += (-Momentum) * dWprev[i] + (1 + Momentum) * V;
                dWprev[i] = V;
            }
        }

        Scalar LearningRate;
        Scalar WeightDecay;
        Scalar Momentum;
    };

}  // namespace tiny_dnn
