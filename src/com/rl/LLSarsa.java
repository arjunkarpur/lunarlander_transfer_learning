package com.rl;

import burlap.behavior.policy.GreedyQPolicy;
import burlap.behavior.policy.Policy;
import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.EpisodeSequenceVisualizer;
import burlap.behavior.singleagent.auxiliary.StateGridder;
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

public class LLSarsa {

    public static void LLSARSA(){

        LunarLanderDomain lld = new LunarLanderDomain();
        Domain domain = lld.generateDomain();
        RewardFunction rf = new LunarLanderRF(domain);
        TerminalFunction tf = new LunarLanderTF(domain);

        State s = LunarLanderDomain.getCleanState(domain, 0);
        LunarLanderDomain.setAgent(s, 0., 5., 0.);
        LunarLanderDomain.setPad(s, 75., 95., 0., 10.);

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
        GradientDescentSarsaLam agent = new GradientDescentSarsaLam(domain, 0.99, vfa, 0.02, 0.5);

        SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, s);
        List<EpisodeAnalysis> episodes = new ArrayList();
        for(int i = 0; i < 5000; i++){
            EpisodeAnalysis ea = agent.runLearningEpisode(env);
            episodes.add(ea);
            //System.out.println(i + ": " + ea.maxTimeStep());
            env.resetEnvironment();
        }

        Visualizer v = LLVisualizer.getVisualizer(lld.getPhysParams());
        new EpisodeSequenceVisualizer(v, domain, episodes);

    }

    public static GradientDescentSarsaLam runExperimentAndGetVFA(int taskID) {
        LunarLanderDomain lld = new LunarLanderDomain();
        Domain domain = lld.generateDomain();
        RewardFunction rf = new LunarLanderRF(domain);
        TerminalFunction tf = new LunarLanderTF(domain);

        State s = null;
        if (taskID == 1) {
            s = LunarLanderDomain.getCleanState(domain, 0);
            LunarLanderDomain.setAgent(s, 0., 5., 30.);
            LunarLanderDomain.setPad(s, 75., 95., 0., 10.);
        } else if (taskID == 2) {
            s = LunarLanderDomain.getCleanState(domain, 1);
            LunarLanderDomain.setAgent(s, 0., 5., 30.);
            LunarLanderDomain.setObstacle(s, 0, 30., 50, 20, 40);
            LunarLanderDomain.setPad(s, 50, 100., 0., 1.);
        }

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
        GradientDescentSarsaLam agent = new GradientDescentSarsaLam(domain, 0.99, vfa, 0.02, 0.5);

        SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, s);
        List<EpisodeAnalysis> episodes = new ArrayList();
        for(int i = 0; i < 5000; i++){
            EpisodeAnalysis ea = agent.runLearningEpisode(env);
            episodes.add(ea);
            System.out.println(i + ": " + ea.maxTimeStep());
            env.resetEnvironment();
        }

        Visualizer v = LLVisualizer.getVisualizer(lld.getPhysParams());
        new EpisodeSequenceVisualizer(v, domain, episodes);


        if (taskID == 2) {
            s = LunarLanderDomain.getCleanState(domain, 1);
            LunarLanderDomain.setAgent(s, 0., 5., 30.);
            LunarLanderDomain.setObstacle(s, 0, 30., 50, 20, 40);
            LunarLanderDomain.setPad(s, 50, 100., 0., 1.);

            System.out.println("initial state approximation: " + vfa.getStateValue(s).predictedValue);
        }


        return agent;
    }

    public static RewardFunction transferRewardFunction(GradientDescentSarsaLam vfaOne, GradientDescentSarsaLam vfaTwo) {
        LunarLanderDomain lld = new LunarLanderDomain();
        Domain domain = lld.generateDomain();
        RewardFunction rf = new LunarLanderRF(domain);

        State s = LunarLanderDomain.getCleanState(domain, 1);
        LunarLanderDomain.setAgent(s, 0., 5., 30.);
        LunarLanderDomain.setObstacle(s, 0, 30., 50, 20, 40);
        LunarLanderDomain.setPad(s, 75., 95., 0., 10.);

        ShapedRewardFunction shapedRF = new ShapedRewardFunction(rf) {
            @Override
            public double additiveReward(State s, GroundedAction a, State sprime) {
                double potential1 = .99*vfaOne.value(sprime) - vfaOne.value(s);
                double potential2 = .99*vfaTwo.value(sprime) - vfaTwo.value(s);


                return potential1 + potential2;

//                double x2 = sprime.getFirstObjectOfClass(LunarLanderDomain.AGENTCLASS).getNumericValForAttribute(LunarLanderDomain.XATTNAME);
//                double y2 = sprime.getFirstObjectOfClass(LunarLanderDomain.AGENTCLASS).getNumericValForAttribute(LunarLanderDomain.YATTNAME);
//
//                double x1 = s.getFirstObjectOfClass(LunarLanderDomain.AGENTCLASS).getNumericValForAttribute(LunarLanderDomain.XATTNAME);
//                double y1 = s.getFirstObjectOfClass(LunarLanderDomain.AGENTCLASS).getNumericValForAttribute(LunarLanderDomain.YATTNAME);
//
//                double x3 = s.getFirstObjectOfClass(LunarLanderDomain.PADCLASS).getNumericValForAttribute(LunarLanderDomain.LATTNAME);
//                double y3 = s.getFirstObjectOfClass(LunarLanderDomain.PADCLASS).getNumericValForAttribute(LunarLanderDomain.TATTNAME);
//
//                return 0 - (.99*Math.sqrt(Math.pow(x3-x2, 2) + Math.pow(y3-y2, 2)) - Math.sqrt(Math.pow(x3 - x1, 2) + Math.pow(y3 - y1, 2)));
            }
        };
        return shapedRF;
    }

    public static void learnUsingShapedRF(RewardFunction rf) {
        LunarLanderDomain lld = new LunarLanderDomain();
        Domain domain = lld.generateDomain();
        TerminalFunction tf = new LunarLanderTF(domain);

        State s = LunarLanderDomain.getCleanState(domain, 1);
        LunarLanderDomain.setAgent(s, 0., 5., 30.);
        LunarLanderDomain.setObstacle(s, 0, 30., 50, 20, 40);
        LunarLanderDomain.setPad(s, 75, 95., 0., 10.);

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
        GradientDescentSarsaLam agent = new GradientDescentSarsaLam(domain, 0.99, vfa, 0.02, 0.5);

        SimulatedEnvironment env = new SimulatedEnvironment(domain, rf, tf, s);

        List<EpisodeAnalysis> episodes = new ArrayList();
        for(int i = 0; i < 1000; i++){
            EpisodeAnalysis ea = agent.runLearningEpisode(env);
            episodes.add(ea);
            System.out.println(i + ": " + ea.maxTimeStep());
            env.resetEnvironment();
        }

        Visualizer v = LLVisualizer.getVisualizer(lld.getPhysParams());
        new EpisodeSequenceVisualizer(v, domain, episodes);


    }


    public static void main(String[] args) {
        GradientDescentSarsaLam vfaOne = runExperimentAndGetVFA(1);
        GradientDescentSarsaLam vfaTwo = runExperimentAndGetVFA(2);

        RewardFunction transferedRF = transferRewardFunction(vfaOne, vfaTwo);
        learnUsingShapedRF(transferedRF);

        //LLSARSA();

    }

}
