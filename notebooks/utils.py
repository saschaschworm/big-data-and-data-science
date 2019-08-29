# Python Packages
from collections import deque
import itertools
from timeit import default_timer as timer
from typing import Optional, Union

# Third-Party Packages
from graphviz import Source
from matplotlib import cm
from matplotlib import animation
from matplotlib.animation import FileMovieWriter
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import pandas as pd
import pygraphviz as pgv
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree as ctree


class AnimationFileWriter(FileMovieWriter):
    """
    https://stackoverflow.com/a/51778313/5953266
    """
    supported_formats = ['png', 'jpeg', 'bmp', 'svg', 'pdf']

    def __init__(self, *args, extra_args=None, **kwargs):
        super().__init__(*args, extra_args=(), **kwargs)

    def setup(self, fig, dpi, frame_prefix):
        super().setup(fig, dpi, frame_prefix, clear_temp=False)
        self.fname_format_str = '%s%%d.%s'
        self.temp_prefix, self.frame_format = self.outfile.split('.')
        self.temp_prefix = '../slides/figures/' + self.temp_prefix 

    def grab_frame(self, **savefig_kwargs):
        '''
        Grab the image information from the figure and save as a movie frame.
        All keyword arguments in savefig_kwargs are passed on to the 'savefig'
        command that saves the figure.
        '''
        with self._frame_sink() as myframesink:
            self.fig.savefig(myframesink, format=self.frame_format,
                             dpi=self.dpi, **savefig_kwargs)

    def finish(self):
        self._frame_sink().close()
        

def plot_confusion_matrix(cm, classes, normalize=False, 
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] 

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    
class GradientDescentRegressor(object):
    def __init__(
            self,
            method: str = 'default',
            loss: str = 'rmse',
            learning_rate: Union[int, float] = 0.0001,
            tol: float = 0.01,
            max_iter: Optional[int] = 1000,
            random_state: Optional[int] = None,
            verbose: int = 0
    ) -> None:

        if method not in ['default', 'stochastic']:
            raise ValueError(f"Method must be either 'default' or 'stochastic'")
        
        if loss not in ['mse', 'rmse']:
            raise ValueError(f"Loss function must be either 'mse' or 'rmse'")
        
        if not (isinstance(learning_rate, float) or isinstance(learning_rate, int)):
            raise TypeError(f'Learning rate must be either of type float or int')
        
        if not isinstance(tol, float):
            raise TypeError(f'Tolerance must be of type float')
            
        if max_iter:
            if not isinstance(max_iter, int):
                raise TypeError(f'Max iterations must be either None or of type int')
        if max_iter < 1:
            raise ValueError(f'Max iterations must be positive and at least 1')
            
        if verbose not in [0, 1]:
            raise ValueError(f'Verbose mode must be either 0 (off) or 1 (on)')
        
        if random_state:
            if not isinstance(random_state, int):
                raise TypeError(f'Random state must be either None or of type int')
            np.random.seed(random_state)
        
        self.__method = method
        self.__loss = loss        
        self.__learning_rate = learning_rate
        self.__tol = tol      
        self.__max_iter = max_iter
        self.__verbose = verbose
        self.__random_state = random_state
        
        self.__X = None
        self.__y = None
        self.__n_features = None
        self.__parameters = None
        self.__start_time = None
        self.__end_time = None

        self.loss_history = []
        self.parameter_history = []
        self.iterations = 0
        self.score = 0
        self.duration = 0
        self.parameters = []

    def __repr__(self) -> str:
        return f'GradientDescentRegressor(method={self.__method}, loss={self.__loss}, ' \
               f'learning_rate={self.__learning_rate}, tol={self.__tol}, ' \
               f'max_iter={self.__max_iter}, random_state={self.__random_state}, ' \
               f'verbose={self.__verbose})'

    @staticmethod
    def rmse(y_true, y_pred):
        return np.sqrt(np.sum(np.square(y_true - y_pred)) / len(y_true))
    
    @staticmethod
    def mse(y_true, y_pred):
        return np.sum(np.square(y_true - y_pred)) / (2 * len(y_true))
    
    @staticmethod
    def smape(y_true, y_pred):
        absolute_difference = np.abs(y_pred - y_true)
        absolute_sum = np.abs(y_true) + np.abs(y_pred)
        return (100/len(y_true) * np.sum(2 * absolute_difference / absolute_sum)) / 100

    def __hypothesis(self, X, parameters):
        X = np.insert(X, 0, 1, axis=1)
        return X.dot(parameters)
    
    def __cost_function(self, X, y, parameters):
        if self.__loss == 'rmse':
            return self.rmse(y, self.__hypothesis(X, parameters))
        elif self.__loss == 'mse':
            return self.mse(y, self.__hypothesis(X, parameters))
    
    def __fit_gd(self):
        error = self.__hypothesis(self.__X, self.__parameters) - self.__y
        loss = self.__cost_function(self.__X, self.__y, self.__parameters)
        
        self.loss_history.append(loss)
        self.parameter_history.append(self.__parameters)
        self.iterations += 1
        
        if self.__verbose > 0:
            print(f'Iteration={self.iterations}, Parameters={np.round(self.__parameters, 2)}, '
                  f'Loss={loss:.2f}')
        
        while abs(loss - self.score) > self.__tol:
            self.score = loss
            
            gradient = self.__X.T.dot(error) / self.__y.size
            self.__parameters = self.__parameters - self.__learning_rate * gradient

            error = self.__hypothesis(self.__X, self.__parameters) - self.__y
            loss = self.__cost_function(self.__X, self.__y, self.__parameters)

            self.loss_history.append(loss)
            self.parameter_history.append(self.__parameters)
            self.iterations += 1
            
            if self.__verbose > 0:
                print(f'Iteration={self.iterations}, Parameters={np.round(self.__parameters, 2)}, '
                      f'Loss={loss:.2f}')
            
            if self.__max_iter:
                if self.iterations == self.__max_iter:
                    break
                
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientDescentRegressor':
        if not isinstance(X, np.ndarray):
            raise TypeError(f'X must be of type ndarray')
        if len(X.shape) < 2:
            raise ValueError(f'X must be a matrix')
        if not isinstance(y, np.ndarray):
            raise TypeError(f'y must be of type ndarray')
        if len(y.shape) > 1:
            raise ValueError(f'y must be a vector')
        
        self.__start_time = timer()
        
        method = 'gradient descent' if self.__method == 'default' else 'stochastic dradient descent'
        
        self.__X = X
        self.__y = y
        self.__n_features = self.__X.shape[1] + 1
        self.__parameters = np.random.randint(0, 100, self.__n_features)  

        if self.__verbose > 0:
            print(f'Starting parameter optimization using {method}...')
        
        if self.__method == 'default':
            self.__fit_gd()
        elif self.__method == 'stochastic':
            raise NotImplementedError(f'Stochastic gradient descent has not been implemented yet')
            
        self.loss_history = np.array(self.loss_history)
        self.parameter_history = np.array(self.parameter_history)
        self.parameters = self.__parameters
            
        self.__end_time = timer()
        self.duration = self.__end_time - self.__start_time
        
        if self.__verbose > 0:
            method = 'gradient descent' \
                if self.__method == 'default' \
                else 'stochastic dradient descent'
            print(f'Parameter optimization with {method} finished in {self.duration:.2f}s.')
        
        return self
    
    def predict(self, X: np.ndarray, parameters: Optional[np.ndarray] = None) -> np.ndarray:
        if 'iterations' not in self.__dict__:
            raise BaseException('You cannot predict when the model has not optmized the parameters')
        
        if not isinstance(X, np.ndarray):
            raise TypeError(f'X must be of type ndarray')
        if len(X.shape) < 2:
            raise ValueError(f'X must be a matrix')
        if not isinstance(parameters, np.ndarray):
            parameters = self.__parameters
            
        return self.__hypothesis(X, parameters)
    
    def create_animation(
            self,
            x_label: Optional[str] = None,
            y_label: Optional[str] = None,
            dpi=72
    ) -> animation.FuncAnimation:

        if 'iterations' not in self.__dict__:
            raise BaseException('You cannot visualize when the model has not optmized the '
                                'parameters')
        if self.__n_features - 1 > 1:
            raise NotImplementedError('Visualization for multiple regression has not been '
                                      'implemented yet')
        
        max_x, min_x = np.round(np.max(self.__X), 0) + 2.5, np.round(np.min(self.__X), 0) - 2.5
        max_y, min_y = np.round(np.max(self.__y), 0) + 20.5, np.round(np.min(self.__y), 0) - 20.5
        x_range = np.linspace(0, max_x, 100)[:, np.newaxis]
        
        parameters = np.linspace(np.max(self.parameter_history[:,0]) * -1,
                                 np.max(self.parameter_history[:,0]) + 20, 40), \
                     np.linspace(np.max(self.parameter_history[:,1]) * -1,
                                 np.max(self.parameter_history[:,1]) + 20, 40)
        pX, pY = np.meshgrid(parameters[0], parameters[1])
        pZ = np.array([self.__cost_function(self.__X, self.__y, parameters)
                       for parameters in zip(np.ravel(pX), np.ravel(pY))]).reshape(pX.shape)
        
        fig = plt.figure(figsize=(16, 6), dpi=dpi)
        ax1, ax2, ax3, ax4 = fig.add_subplot(221), fig.add_subplot(222), fig.add_subplot(223), \
                             fig.add_subplot(224, projection='3d')

        points1, = ax1.plot(
            self.parameter_history[:, 0][0], self.loss_history[0], marker='o', ls='')
        points2, = ax2.plot(
            self.parameter_history[:, 1][0], self.loss_history[0], marker='o', ls='')
        line1, = ax3.plot(
            x_range, self.predict(x_range, self.parameter_history[0]), label='Regression Model')
        line2, = ax4.plot(
            self.parameter_history[:1,0], self.parameter_history[:1,1], self.loss_history[0] ,
            markerfacecolor='b', markeredgecolor='b', marker='.', markersize=5);
        line3, = ax4.plot(
            self.parameter_history[:1,0], self.parameter_history[:1,1], 0, markerfacecolor='b',
            markeredgecolor='b', marker='.', markersize=5);
        
        def update(idx):
            points1.set_data(self.parameter_history[idx][0], self.loss_history[idx])
            points2.set_data(self.parameter_history[idx][1], self.loss_history[idx])   
            
            line1.set_ydata(self.predict(x_range, self.parameter_history[idx]))
            
            line2.set_data((self.parameter_history[:idx + 1,0], self.parameter_history[:idx + 1,1]))
            line2.set_3d_properties(self.loss_history[:idx + 1])
            
            line3.set_data((self.parameter_history[:idx + 1,0], self.parameter_history[:idx + 1,1]))
            line3.set_3d_properties(0)
            
            title = f'Iteration: {idx}, $\\theta_0={self.parameter_history[idx][0]:.2f}$, ' \
                    f'$\\theta_1={self.parameter_history[idx][1]:.2f}$, ' \
                    f'$J(\\theta)={self.loss_history[idx]:.2f}$'
            fig.suptitle(title, y=1, fontsize=18)
        
        ax1.plot(self.parameter_history[:,0], self.loss_history)
        ax1.set_xlabel("$\\theta_0$")
        ax1.set_ylabel("$J(\\theta)$")
        ax1.grid(linestyle=':')
        
        ax2.plot(self.parameter_history[:,1], self.loss_history)
        ax2.set_xlabel("$\\theta_1$")
        ax2.set_ylabel("$J(\\theta)$")
        ax2.grid(linestyle=':')

        ax3.scatter(self.__X, self.__y, label="Observations", s=20)
        ax3.set_xlabel(x_label)
        ax3.set_ylabel(y_label)
        ax3.set_xlim(min_x, max_x)
        ax3.set_ylim(min_y, max_y)
        ax3.legend(loc="upper left");
        ax3.grid(linestyle=':')
        
        ax4.view_init(elev=20., azim=-10)        
        ax4.plot_surface(pX, pY, pZ, rstride=1, cstride=1, cmap=cm.coolwarm, alpha=0.8)
        ax4.contour(pX, pY, pZ, 20, alpha=0.2, offset=0, stride=30)
        ax4.plot(
            [self.parameters[0]], [self.parameters[1]], [self.score] , markerfacecolor='c',
            markeredgecolor='c', marker='o', markersize=10);
        ax4.set_xlabel('$\\theta_0$')
        ax4.set_ylabel('$\\theta_1$')
        ax4.set_zlabel('$J(\\theta)$')
        ax4.dist = 7
        
        plt.tight_layout(pad=5, h_pad=0, w_pad=0)
        plt.close() 
        
        return animation.FuncAnimation(fig, update, frames=self.iterations, interval=200)
    
    
class AABB:
    """
    Axis-aligned bounding box
    https://stackoverflow.com/a/43935462/5953266
    """
    def __init__(self, n_features):
        self.limits = np.array([[-np.inf, np.inf]] * n_features)

    def split(self, f, v):
        left = AABB(self.limits.shape[0])
        right = AABB(self.limits.shape[0])
        left.limits = self.limits.copy()
        right.limits = self.limits.copy()

        left.limits[f, 1] = v
        right.limits[f, 0] = v

        return left, right


def tree_bounds(tree, n_features=None):
    """
    Compute final decision rule for each node in tree
    https://stackoverflow.com/a/43935462/5953266
    """
    if n_features is None:
        n_features = np.max(tree.feature) + 1
    aabbs = [AABB(n_features) for _ in range(tree.node_count)]
    queue = deque([0])
    while queue:
        i = queue.pop()
        l = tree.children_left[i]
        r = tree.children_right[i]
        if l != ctree.TREE_LEAF:
            aabbs[l], aabbs[r] = aabbs[i].split(tree.feature[i], tree.threshold[i])
            queue.extend([l, r])
    return aabbs


def decision_areas(tree_classifier, maxrange, x=0, y=1, n_features=None):
    """
    Extract decision areas.

    tree_classifier: Instance of a sklearn.tree.DecisionTreeClassifier
    maxrange: values to insert for [left, right, top, bottom] if the interval is open (+/-inf) 
    x: index of the feature that goes on the x axis
    y: index of the feature that goes on the y axis
    n_features: override autodetection of number of features
    
    https://stackoverflow.com/a/43935462/5953266
    """
    tree = tree_classifier.tree_
    aabbs = tree_bounds(tree, n_features)

    rectangles = []
    for i in range(len(aabbs)):
        if tree.children_left[i] != ctree.TREE_LEAF:
            continue
        l = aabbs[i].limits
        r = [l[x, 0], l[x, 1], l[y, 0], l[y, 1], np.argmax(tree.value[i])]
        rectangles.append(r)
    rectangles = np.array(rectangles)
    rectangles[:, [0, 2]] = np.maximum(rectangles[:, [0, 2]], maxrange[0::2])
    rectangles[:, [1, 3]] = np.minimum(rectangles[:, [1, 3]], maxrange[1::2])
    return rectangles

def plot_areas(rectangles):
    """
    https://stackoverflow.com/a/43935462/5953266
    """
    for rect in rectangles:
        color = ['b', 'r'][int(rect[4])]
        print(rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])
        rp = Rectangle([rect[0], rect[2]], 
                       rect[1] - rect[0], 
                       rect[3] - rect[2], color=color, alpha=0.3)
        plt.gca().add_artist(rp)

        
class ID3(object):
    def __init__(self, target_variable, condition_true='yes', condition_false='no'):
        self.target_variable = target_variable
        self.condition_true = condition_true
        self.condition_false = condition_false
        self.tree = {}
        self.paths = []
        self.nodes = []
        self.edges = []
    
    def _calculate_entropy(self, subset):
        instances_true, instances_false = subset[subset[self.target_variable] == self.condition_true], subset[subset[self.target_variable] == self.condition_false]
        n_instances, n_instances_true, n_instances_false = len(subset), len(instances_true), len(instances_false)
        proportion_true, proportion_false = n_instances_true / n_instances, n_instances_false / n_instances
        if proportion_true == 1.0 or proportion_false == 1.0:
            return 0, proportion_true
        return  (-proportion_true * np.log2(proportion_true)) + (-proportion_false * np.log2(proportion_false)), proportion_true
    
    def _construct_paths(self, subset, path='', attribute='', node=''):    
        path += node + '/'
        if path != '/':
            path += attribute + '/'
        
        n_subset_instances = len(subset)
        subset_entropy, subset_proportion_true = self._calculate_entropy(subset)
        features, information_gains = list(subset.loc[:, subset.columns != self.target_variable]), []

        if subset_entropy == 0.0:
            path += 'yes' if subset_proportion_true == 1.0 else 'no'
            self.paths.append(path)
            return
        for feature in features:
            information_gain = subset_entropy
            for attribute in subset[feature].unique():
                attribute_subset = subset[subset[feature] == attribute][[feature, self.target_variable]]
                n_attribute_subset_instances = len(attribute_subset)
                attribute_subset_entropy, attribute_subset_proportion_true = self._calculate_entropy(attribute_subset)
                information_gain -= n_attribute_subset_instances / n_subset_instances * attribute_subset_entropy
            information_gains.append(information_gain)
        best_split_feature = features[information_gains.index(max(information_gains))]
        best_split_feature_attributes = subset[best_split_feature].unique()
        for best_split_feature_attribute in best_split_feature_attributes:
            self._construct_paths(subset[subset[best_split_feature] == best_split_feature_attribute].loc[:, subset.columns != best_split_feature], path=path, attribute=best_split_feature_attribute, node=best_split_feature)
    
    def _construct_graph(self, tree, parent=None):
        for key, value in tree.items():
            if isinstance(value, dict):
                self.nodes.append(key)
                if parent:
                    self.edges.append((parent, key))
                self._construct_graph(value, key)
            else:
                self.nodes.append(key)
                self.edges.append((parent, key))

    def _construct_tree(self):
        for p in self.paths:
            parts = p.split('/')
            branch = self.tree
            for part in parts[1:-1]:
                branch = branch.setdefault(part, {})
            branch[parts[-1]] = 1 + branch.get(parts[-1], 0)
        self._construct_graph(self.tree)

    def fit(self, subset):
        self._construct_paths(subset)
        self._construct_tree()
        
    def print_graph(self, level=None):
        graph = pgv.AGraph(strict=False, directed=True)
        graph.graph_attr['rankdir'] = 'TB'
        graph.graph_attr['dpi'] =  72
        graph.graph_attr['fontname'] = 'Helvetica'
        graph.node_attr['shape'] = 'box'
        graph.node_attr['fontname'] = 'Helvetica'
        graph.edge_attr['fontname'] = 'Helvetica'

        for idx, node in enumerate(self.nodes[:level + 1]):
            if idx % 2 == 0:
                label = '<<font point-size="10" face="Helvetica-Bold">{}</font>>'.format(node)
                shape = 'box'
                fixedsize = False
            else:
                label = '<<font point-size="8" face="Helvetica">{}</font>>'.format(node)
                shape = 'box'
                fixedsize = True
            graph.add_node(node, label=label, shape=shape, width=0.5, fixedsize=fixedsize)

        for edge in self.edges[:level]:
            graph.add_edge(edge[0], edge[1])

        return Source(graph) 
    
    
def plot_tree(nodes, nodes_idx, edges, marked_nodes, leaves=[], filename=None):
    graph = pgv.AGraph(strict=False, directed=True)
    graph.graph_attr['rankdir'] = 'LR'
    graph.graph_attr['dpi'] =  72
    graph.graph_attr['fontname'] = 'Helvetica'
    graph.node_attr['shape'] = 'box'
    graph.node_attr['fontname'] = 'Helvetica'
    graph.edge_attr['fontname'] = 'Helvetica'
    
    for idx, node_idx in enumerate(nodes_idx):
        fillcolor = 'white'
        fontcolor = 'black'
        
        if idx in marked_nodes:
            label = '<<font point-size="10" face="Helvetica-Bold">{}</font>>'.format(nodes[idx])
            shape = 'box'
            fixedsize = False
        else:
            fillcolor = 'gray32'
            fontcolor = 'white'
            label = '<<font point-size="7" face="Helvetica-Bold">{}</font>>'.format(nodes[idx])
            shape = 'box'
            fixedsize = True
            
        if idx in leaves:
            if nodes[idx] == 'YES':
                fillcolor = 'darkgreen'
                fontcolor = 'white'
            elif nodes[idx] == 'NO':
                fillcolor = 'orangered3'
                fontcolor = 'white'

        graph.add_node(node_idx, label=label, shape=shape, width=0.5, fixedsize=fixedsize, fillcolor=fillcolor, fontcolor=fontcolor, style='filled')
        
    for edge in edges:
        graph.add_edge(edge[0], edge[1])
        
    if filename:
        graph.draw(filename, prog='dot')

    return Source(graph)
