package com.rl;

import burlap.behavior.singleagent.EpisodeAnalysis;
import burlap.behavior.singleagent.auxiliary.performance.LearningAlgorithmExperimenter;
import burlap.behavior.singleagent.auxiliary.performance.PerformanceMetric;
import burlap.behavior.singleagent.auxiliary.performance.TrialMode;
import burlap.behavior.singleagent.learning.LearningAgent;
import burlap.behavior.singleagent.learning.LearningAgentFactory;
import burlap.behavior.singleagent.learning.tdmethods.vfa.GradientDescentSarsaLam;
import burlap.behavior.singleagent.shaping.ShapedRewardFunction;
import burlap.behavior.singleagent.vfa.ValueFunctionApproximation;
import burlap.behavior.singleagent.vfa.cmac.CMACFeatureDatabase;
import burlap.domain.singleagent.lunarlander.LunarLanderDomain;
import burlap.domain.singleagent.lunarlander.LunarLanderRF;
import burlap.domain.singleagent.lunarlander.LunarLanderTF;
import burlap.oomdp.core.Domain;
import burlap.oomdp.core.TerminalFunction;
import burlap.oomdp.core.states.State;
import burlap.oomdp.singleagent.GroundedAction;
import burlap.oomdp.singleagent.RewardFunction;
import burlap.oomdp.singleagent.environment.SimulatedEnvironment;

import java.util.ArrayList;
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

        target.setRf(rf);
        LearningAgentFactory agent = getAgentFactory("target task", target);

        LearningAlgorithmExperimenter exp = new LearningAlgorithmExperimenter(target, 10, 6000, agent);
        exp.setUpPlottingConfiguration(800, 800, 2, 1000, TrialMode.MOSTRECENTANDAVERAGE, PerformanceMetric.AVERAGEEPISODEREWARD);
        exp.startExperiment();

        //exp.writeEpisodeDataToCSV("expDatTransfer");

        List<EpisodeAnalysis> episodes = new ArrayList();
        EpisodeAnalysis ea;
        for(int i = 0; i < 10000; i++){
            ea = agent.generateAgent().runLearningEpisode(target);
            episodes.add(ea);
            System.out.println(i + ": " + ea.maxTimeStep());
            target.resetEnvironment();
        }
        /*
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


        // Set up environment with no obstacles
        SimulatedEnvironment source0 = getLanderEnvironment(null, new LLRectangle(75.,95.,0.,10.), new double[]{5.,30.});
        LearningAgentFactory agent0 = getAgentFactory("source_task_0", source0);

        // Set up an environment with obstacles in the way
        LLRectangle[] obstacles = new LLRectangle[] {new LLRectangle(30.,50.,20.,40.)};
        SimulatedEnvironment source1 = getLanderEnvironment(obstacles, new LLRectangle(50.,100.,0.,1.), new double[]{5.,30.});
        LearningAgentFactory agent1 = getAgentFactory("source_task_1", source1);

        // Target task domain
        SimulatedEnvironment targetDomain = getLanderEnvironment(obstacles, new LLRectangle(75.,95.,0.,10.), new double[]{5.,30.});

        // Learn in source tasks
        LearningAgent agent0_gen = agent0.generateAgent();
        runLearning(agent0_gen, source0, 6000);
        LearningAgent agent1_gen = agent1.generateAgent();
        runLearning(agent1_gen, source1, 6000);

        // Run learning without transfer
        LunarLanderDomain lld = new LunarLanderDomain();
        Domain domain = lld.generateDomain();
        RewardFunction baseRF = new LunarLanderRF(domain);
        learnUsingShapedRF(baseRF, targetDomain);

        // Learn with transfer
        GradientDescentSarsaLam[] sarsas = {(GradientDescentSarsaLam)agent0_gen, (GradientDescentSarsaLam) agent1_gen};
        RewardFunction transferedRF = transferRewardFunction(sarsas);
        learnUsingShapedRF(transferedRF, targetDomain);

    }

}
