package com.rl;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateGridder;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.PerformancePlotter;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.lspi.LSPI;
import burlap.behavior.singleagent.learning.lspi.SARSCollector;
import burlap.behavior.singleagent.learning.lspi.SARSData;
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam;
import burlap.behavior.singleagent.planning.stochastic.sparsesampling.SparseSampling;
import burlap.behavior.singleagent.shaping.ShapedRewardFunction;
import burlap.behavior.singleagent.vfa.ValueFunctionApproximation;
import burlap.behavior.singleagent.vfa.cmac.CMACFeatureDatabase;
import burlap.behavior.singleagent.vfa.common.ConcatenatedObjectFeatureVectorGenerator;
import burlap.behavior.singleagent.vfa.fourier.FourierBasis;
import burlap.behavior.singleagent.vfa.rbf.DistanceMetric;
import burlap.behavior.singleagent.vfa.rbf.RBFFeatureDatabase;
import burlap.behavior.singleagent.vfa.rbf.functions.GaussianRBF;
import burlap.behavior.singleagent.vfa.rbf.metrics.EuclideanDistance;
import burlap.domain.singleagent.cartpole.InvertedPendulum;
import burlap.domain.singleagent.cartpole.InvertedPendulumVisualizer;
import burlap.domain.singleagent.lunarlander.LLVisualizer;
import burlap.domain.singleagent.lunarlander.LunarLanderDomain;
import burlap.domain.singleagent.lunarlander.LunarLanderRF;
import burlap.domain.singleagent.lunarlander.LunarLanderTF;
import burlap.domain.singleagent.mountaincar.MCRandomStateGenerator;
import burlap.domain.singleagent.mountaincar.MountainCar;
import burlap.domain.singleagent.mountaincar.MountainCarVisualizer;
import burlap.oomdp.auxiliary.StateGenerator;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.common.GoalBasedRF;
import burlap.oomdp.singleagent.common.VisualActionObserver;
import burlap.oomdp.singleagent.environment.EnvironmentServer;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;
import burlap.oomdp.statehashing.SimpleHashableStateFactory;
import burlap.oomdp.visualizer.Visualizer;
import com.sun.prism.paint.Gradient;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class LLRectangle {
    double l, r, b, t;
    public LLRectangle(double left, double right, double bottom, double top) {
        l = left;
        r = right;
        b = bottom;
        t = top;
    }

    public double left() { return l; }
    public double right() { return r; }
    public double top() { return t; }
    public double bottom() { return b; }
}
public class LLSarsa {

    public static LearningAgentFactory getAgentFactory(String agentName, SimulatedEnvironment env) {

        LearningAgentFactory transferLearningFactory = new LearningAgentFactory() {
            @Override
            public String getAgentName() {
                return agentName;
            }

            @Override
            public LearningAgent generateAgent() {

                Domain domain = env.getDomain();
                LunarLanderDomain lld = new LunarLanderDomain();
                //---------- Set up linear approximation method --------------
                int nTilings = 5;
                CMACFeatureDatabase cmac = new CMACFeatureDatabase(nTilings,
                        CMACFeatureDatabase.TilingArrangement.RANDOMJITTER);
                double resolution = 10.;

                double angleWidth = 2 * lld.getAngmax() / resolution;
                double xWidth = (lld.getXmax() - lld.getXmin()) / resolution;
                double yWidth = (lld.getYmax() - lld.getYmin()) / resolution;
                double velocityWidth = 2 * lld.getVmax() / resolution;

                cmac.addSpecificationForAllTilings(LunarLanderDomain.AGENTCLASS,
                        domain.getAttribute(LunarLanderDomain.AATTNAME),
                        angleWidth);
                cmac.addSpecificationForAllTilings(LunarLanderDomain.AGENTCLASS,
                        domain.getAttribute(LunarLanderDomain.XATTNAME),
                        xWidth);
                cmac.addSpecificationForAllTilings(LunarLanderDomain.AGENTCLASS,
                        domain.getAttribute(LunarLanderDomain.YATTNAME),
                        yWidth);
                cmac.addSpecificationForAllTilings(LunarLanderDomain.AGENTCLASS,
                        domain.getAttribute(LunarLanderDomain.VXATTNAME),
                        velocityWidth);
                cmac.addSpecificationForAllTilings(LunarLanderDomain.AGENTCLASS,
                        domain.getAttribute(LunarLanderDomain.VYATTNAME),
                        velocityWidth);


                double defaultQ = 0.5;
                ValueFunctionApproximation vfa = cmac.generateVFA(defaultQ/nTilings);

                GradientDescentSarsaLam s = new GradientDescentSarsaLam(domain, .99, vfa, 0.02, 0.5);


                return s;
            }
        };

        return transferLearningFactory;
    }

    public static SimulatedEnvironment getLanderEnvironment(LLRectangle[] obstacles,
                                                          LLRectangle pad, double[] lander) {
        LunarLanderDomain lld = new LunarLanderDomain();
        Domain domain = lld.generateDomain();
        RewardFunction rf = new LunarLanderRF(domain);
        TerminalFunction tf = new LunarLanderTF(domain);

        State s = LunarLanderDomain.getCleanState(domain, obstacles != null ? obstacles.length : 0);
        LunarLanderDomain.setAgent(s, 0., lander[0], lander[1]);
        if(obstacles != null) {
            for(int i = 0; i < obstacles.length; ++i) {
                LunarLanderDomain.setObstacle(s, i, obstacles[i].left(), obstacles[i].right(),
                        obstacles[i].bottom(), obstacles[i].top());
            }
        }

        LunarLanderDomain.setPad(s, pad.left(), pad.right(), pad.bottom(), pad.top());

        SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, s);
        return env;
    }

    public static RewardFunction transferRewardFunction(GradientDescentSarsaLam[] sarsas) {
        LunarLanderDomain lld = new LunarLanderDomain();
        Domain domain = lld.generateDomain();
        RewardFunction rf = new LunarLanderRF(domain);

        ShapedRewardFunction shapedRF = new ShapedRewardFunction(rf) {
            @Override
            public double additiveReward(State s, GroundedAction a, State sprime) {
                double potential = 0.;
                for(int i = 0; i < sarsas.length; ++i) {
                    potential += .99*sarsas[i].value(sprime) - sarsas[i].value(s);
                }

                return potential;
            }
        };
        return shapedRF;
    }


    public static void learnUsingShapedRF(RewardFunction rf, SimulatedEnvironment target) {

        LLRectangle[] obstacles = new LLRectangle[] {new LLRectangle(30.,50.,20.,40.)};
        target.setRf(rf);
        LearningAgentFactory agent = getAgentFactory("target task", target);

        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(target, 10, 6000, agent);
        exp.setUpPlottingConfiguration(800, 800, 2, 1000, TrialMode.MOSTRECENTANDAVERAGE, PerformanceMetric.CUMULTAIVEREWARDPEREPISODE);
        exp.startExperiment();


        /*
        List<EpisodeAnalysis> episodes = new ArrayList();
        EpisodeAnalysis ea;
        for(int i = 0; i < 10000; i++){
            ea = agent.generateAgent().runLearningEpisode(target);
            episodes.add(ea);
            System.out.println(i + ": " + ea.maxTimeStep());
            target.resetEnvironment();
        }
        Visualizer v = LLVisualizer.getVisualizer(new LunarLanderDomain().getPhysParams());
        new EpisodeSequenceVisualizer(v, target.getDomain(), episodes);*/


    }

    public static void runLearning(LearningAgent agent, SimulatedEnvironment env, int numEpisodes) {

        LunarLanderDomain lld = new LunarLanderDomain();
        List<EpisodeAnalysis> episodes = new ArrayList();


        for(int i = 0; i < numEpisodes; i++){

            EpisodeAnalysis ea = agent.runLearningEpisode(env);

            episodes.add(ea);
            env.resetEnvironment();
            System.out.println(i + ": " + ea.maxTimeStep());

        }



        /*Visualizer v = LLVisualizer.getVisualizer(lld.getPhysParams());
        new EpisodeSequenceVisualizer(v, env.getDomain(), episodes);*/
    }




    public static void main(String[] args) {


        int NUM_OF_TRIALS = 1000;

        // Run simple learning test example
        SimulatedEnvironment envOne = getLanderEnvironment(null, new LLRectangle(75.,95.,0.,10.), new double[]{60.,30.});
        LearningAgentFactory agentOne = getAgentFactory("sourcetask1", envOne);
        LearningAgent sourceAgentOne = agentOne.generateAgent();
        runLearning(sourceAgentOne, envOne, NUM_OF_TRIALS);

        // Transfer to a little bit further back
        SimulatedEnvironment envTwo = getLanderEnvironment(null, new LLRectangle(75.,95.,0.,10.), new double[]{40.,30.});
        LearningAgentFactory agentTwo = getAgentFactory("sourcetask2", envTwo);
        GradientDescentSarsaLam sourceAgentTwo = (GradientDescentSarsaLam) agentTwo.generateAgent();
        GradientDescentSarsaLam[] agents2 = {(GradientDescentSarsaLam) sourceAgentOne};
        RewardFunction transferredTwo = transferRewardFunction(agents2);
        envTwo.setRf(transferredTwo);
        runLearning(sourceAgentTwo, envTwo, NUM_OF_TRIALS);


        // Transfer to even further back
        SimulatedEnvironment envThree = getLanderEnvironment(null, new LLRectangle(75.,95.,0.,10.), new double[]{20.,30.});
        LearningAgentFactory agentThree = getAgentFactory("task3", envThree);
        GradientDescentSarsaLam sourceAgentThree = (GradientDescentSarsaLam) agentThree.generateAgent();
        GradientDescentSarsaLam[] agents3 = {(GradientDescentSarsaLam) sourceAgentTwo};
        RewardFunction transferredThree = transferRewardFunction(agents3);


        // Target task without any transferred learning done
        LunarLanderDomain lld = new LunarLanderDomain();
        Domain domain = lld.generateDomain();
        RewardFunction baseRF = new LunarLanderRF(domain);
        learnUsingShapedRF(baseRF,envThree);

        // Learn using transferred knowledge
        SimulatedEnvironment envThreeTrans = getLanderEnvironment(null, new LLRectangle(75.,95.,0.,10.), new double[]{20.,30.});
        learnUsingShapedRF(transferredThree, envThreeTrans);

    }

}
