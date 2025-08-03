import argparse
import warnings
import os

class ParserArgs(object):
    """
    arguments to be used in the experiment
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.get_general_parser()

    def get_general_parser(self):
        # general settings
        self.parser.add_argument("--seed", type=int, default=13, help="random seed")
        self.parser.add_argument("--round_num", type=int, default=120, help="round number")
        self.parser.add_argument("--C", type=float, default=1.0, help="acive rate: C, numUsersSel=C*numClients")
        self.parser.add_argument("--class_ratio", nargs='*', type=float, default=[0.3, 0.3, 0.3, 0.3, 0.3], help="list of class ratios for noniid task")
        self.parser.add_argument("--local_epochs", nargs='*', type=int, default=[5, 5, 5, 5, 5],
                                 help="local epochs for each task")
        self.parser.add_argument("--num_clients", type=int, default=30, help="number of clients")
        self.parser.add_argument("--exp_num", type=int, default=1, help="experiment number. repeat the same experiment several times")
        self.parser.add_argument("--powerfulCNN", action="store_true", help="Decide if use powerful CNN for EMNIST")
        self.parser.add_argument("--iid_type", nargs='*', default=['noniid', 'noniid', 'noniid', 'iid', 'iid'], help="List of type_iid")
        self.parser.add_argument("--task_type", nargs='*', default=['mnist', 'cifar10', 'fashion_mnist', 'emnist', 'cifar10'], help="List of task types")
        self.parser.add_argument("--alpha", type=float, default=1.0, help="alpha value, alpha-fairness")
        self.parser.add_argument("--alpha2", type=float, default=1.0, help="alpha value, group fairness loss function")
        self.parser.add_argument("--notes", type=str, default='', help="notes to add on the folder name")
        self.parser.add_argument("--bayes_decay", type=float, default=1.0, help="Decay factor for bayesian method")
        self.parser.add_argument("--algo_type", nargs='*', default=['bayesian', 'proposed','random','round_robin'], help="List of algorithms")
        self.parser.add_argument("--data_ratio", type=float, default=1.0, help="data points num=default_num*data_ratio")
        self.parser.add_argument("--cpumodel", action="store_true", help="store model in cpu")
        self.parser.add_argument("--alpha_loss", action="store_true", help="use alpha-fairness loss function, remember to adjust learning rate")
        self.parser.add_argument("--optimal_sampling", action="store_true", help="use optimal sampling")
        self.parser.add_argument("--insist", action="store_true", help="if insist, then experiment will be conducted even if the folder exists")
        self.parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
        self.parser.add_argument("--validation", action="store_true", help="use validation set instead of test set")
        self.parser.add_argument("--unbalance", nargs='*', type=float, default=[1.0, 1.0], help="unbalance[0]% clients will get unbalance[1]% data")
        self.parser.add_argument("--fairness", type=str, default='notfair', help="fairness type")
        self.parser.add_argument("--L", type=float, default=1/0.02, help="learning rate")
        self.parser.add_argument("--approx_optimal", action="store_true", help="act approx optimal")
        self.parser.add_argument("--aggregation_fair", action="store_true", help="act aggregation fairness")
        self.parser.add_argument("--equalP", action="store_true", help="make P equal")
        self.parser.add_argument("--enlarge", action="store_true", help="make P large")
        self.parser.add_argument("--equalP2", action="store_true", help="make P equal, and use optimal Prob")
        self.parser.add_argument("--test", action="store_true", help="test new things")
        self.parser.add_argument("--group_num", type=int, default=1, help="group number")
        self.parser.add_argument("--mse", action="store_true", help="use mse loss function")
        self.parser.add_argument("--client_cpu", nargs='*', type=float, default=[0.25, 0.5, 0.25], help="clients are separated into serveral groups with different "
                                                                                                 "cpu power. straggler, common, expert")
        self.parser.add_argument("--venn_list", nargs='*', type=float, default=[0.6, 0.3, 0.1],
                                 help="clients can handle different tasks")
        self.parser.add_argument("--suboptimal", action="store_true", help="sub optimal")
        self.parser.add_argument("--freshness", action="store_true", help="freshness")
        self.parser.add_argument("--fresh_ratio", type=float, default=0.2, help="subset ratio")
        self.parser.add_argument("--acc", action="store_true", help="use accuracy")
        # fullparticipation
        self.parser.add_argument("--fullparticipation", action="store_true", help="use full participation")
        self.parser.add_argument("--multiM", action="store_true", help="use multiM")
        # slowstart
        self.parser.add_argument("--slowstart", action="store_true", help="use slow start")
        # self.parser.add_argument("--delta", type=float, default=0.0, help="delta: minimum value for probability")
        self.parser.add_argument("--stale", action="store_true", help="use stale updates")
        self.parser.add_argument("--MILA", action="store_true", help="use MILA aggregation")
        self.parser.add_argument("--scaffold", action="store_true", help="use scaffold")
        self.parser.add_argument("--chosenall", action="store_true", help="chosen all")
        self.parser.add_argument("--stale_b0", type=float, default=0, help="initial b0")
        self.parser.add_argument("--stale_b", type=float, default=0, help="decay b")
        # self.parser.add_argument("--use_h0", action="store_true", help="use h0 to approximate new update matrix")
        self.parser.add_argument("--optimal_b", action="store_true", help="decide optimal b")
        # self.parser.add_argument("--fix_robin", type=float, default=0.0,  help="round scale, if 0, ignore, if not 0, go robin")
        # noextra_com
        self.parser.add_argument("--noextra_com", action="store_true", help="no extra computation")
        self.parser.add_argument("--adjustoldVR", action="store_true", help="adjust norm based on different beta")
        self.parser.add_argument("--window", action="store_true", help="use a window filter")
        self.parser.add_argument("--window_size", type=int, default=10, help="window size")
        self.parser.add_argument("--approximation", action="store_true", help="reduce computation")
        self.parser.add_argument("--ubwindow", action="store_true", help="unbiased window aggregation")
        self.parser.add_argument("--ubwindow2", action="store_true", help="unbiased window aggregation")
        self.parser.add_argument("--ubwindow3", action="store_true", help="unbiased window aggregation")
        self.parser.add_argument("--ff", action="store_true", help="fullfill window terms")
        self.parser.add_argument("--LB", type=float, default=0.0, help="lower bound for q")
        self.parser.add_argument("--skipOS", action="store_true", help="skip optimal sampling")
        self.parser.add_argument("--windowSum", action="store_true", help="window slot with summation")
        # window_max
        self.parser.add_argument("--window_max", type=int, default=120, help="number of clients in the window at most")
        # Krank
        self.parser.add_argument("--Krank", action="store_true", help="use Krank")
        self.parser.add_argument("--givenProb", type=float, default=0.0, help="half client p=1+a, other are 1, then normalize")
        # lastLayer
        self.parser.add_argument("--lastLayer", action="store_true", help="use last layer gradient to compute beta")
        self.parser.add_argument("--optimalV", action="store_true", help="obtain the optimal V matrix")
        # efficient V
        self.parser.add_argument("--effV", action="store_true", help="efficient V, without additional computation")
        # force_range, let s be in the range of [0,1]
        self.parser.add_argument("--force_range", action="store_true", help="let s be in the range of [0,1]")
        # V_direct, no difference of two gradients
        self.parser.add_argument("--V_direct", action="store_true", help="no difference of two gradients")
        # randomK, select K clients randomly to do the linear combination
        self.parser.add_argument("--randomK", type=int, default=0, help="select K clients randomly to do the linear combination")
        # randomKglobal
        self.parser.add_argument("--randomKglobal", type=int, default=0, help="select K clients globally to do the linear combination")
        # create a new name for
        self.parser.add_argument("--norm_candidates", action="store_true", help="normalization for candidates in gradient different method")
        # s_slot
        self.parser.add_argument("--s_slot", type=int, default=1,
                                 help="every s_slot rounds, we update the s value")
        # args.lam
        self.parser.add_argument("--lam", type=float, default=0.01, help="lambda for regularization")
        # args.K
        self.parser.add_argument("--K", type=int, default=None, help="number of clients to select for OMP")
        # args.OMP
        self.parser.add_argument("--OMP", action="store_true", help="use OMP for client selection")
        # num_trials
        self.parser.add_argument("--num_trials", type=int, default=None, help="number of trials for each initialization in OMP")
        # three_sample
        self.parser.add_argument("--three_sample", action="store_true", help="use 3-sample initialization for OMP")
        # recentK
        self.parser.add_argument("--recentK", type=int, default=0, help="select recent K clients")

    def get_args(self):
        args = self.parser.parse_args()
        return args