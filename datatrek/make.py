import os
import pickle
import time
import copy
__all__ = ['Node', 'RootNode', 'PickleNode']
class Node(object):
    '''
    A node in make system
    '''
    def __init__(self, target_files=[], dependencies=[]):
        self.dependencies = dependencies
        self.target_files = target_files
    def update(self):
        '''
        The update function to make target_files for dependencies
        '''
        raise NotImplementedError('update function is not implemented')
    def make(self):
        '''
        make this node
        '''
        # make parents
        for parent in self.dependencies:
            parent.make()
        dependency_files = [f for d in self.dependencies for f in d.target_files]
        # NOTE special case
        if len(dependency_files)==0 or len(self.target_files)==0:
            self.update()
            return
        if not all([os.path.exists(f) for f in self.target_files]):
            self.update()
            return
        min_target_time = min(os.path.getmtime(f) for f in self.target_files)
        max_dependency_time = max(os.path.getmtime(f) for f in dependency_files)
        if min_target_time < max_dependency_time:
            self.update()
            return
    def __str__(self):
        dependency_files = [d.target_files for d in self.dependencies]
        return "targets: %s, dependencies: %s" % (self.target_files, str(dependency_files))
class RootNode(Node):
    '''Node for static resource, target_files should always exist'''
    def __init__(self, target_files):
        self.dependencies = []
        self.target_files = target_files
    def update(self):
        for f in self.target_files:
            if not os.path.exists(f):
                raise RuntimeError('root node target %s not exist' % f)
class PhonyNode(Node):
    '''combine multi nodes together'''
    def __init__(self, dependencies):
        self.dependencies = dependencies
        self.target_files = []
    def update(self):
        pass
class VirtualNode(Node):
    '''
    Usage:
    1. combine all dependencies target to one
    2. add new action
    update do nothing
    '''
    def __init__(self, dependencies):
        self.dependencies = dependencies
        self.target_files = [f for d in dependencies for f in d.target_files]
    def make(self):
        '''Just transfer to parents'''
        for parent in self.dependencies:
            parent.make()
    def update(self):
        raise RuntimeError('this method should never be called')
class PickleNode(Node):
    '''
    Node storing self in single pickle file
    '''
    def __init__(self, f, dependencies):
        self.pickle_file = f
        self.target_files = [f]
        self.dependencies = dependencies
        self.loaded_ = False
        self.make_attributes_ = self.__dict__.keys()
        self.make_attributes_.append('make_attributes_')
    def compute(self):
        '''
        do computation and store result in self
        '''
        raise NotImplementedError('compute function is not implemented')
    def update(self):
        print("Update %s" % self.pickle_file)
        start_time = time.time()
        self.compute()
        self.loaded_ = True
        end_time = time.time()
        self.time_used = end_time - start_time
        print("Elapsed time was %g seconds" % self.time_used)
        self.save_data_()
    def save_data_(self):
        # dump self to pickle file without info for make
        # The dependency infos should be recovered in py files
        if not self.loaded_:
            raise RuntimeError('try to save data when data not loaded')
        odict = self.__dict__.copy()
        for k in odict.keys():
            if k in self.make_attributes_:
                del odict[k]
        pickle.dump(odict, open(self.pickle_file, 'wb'), pickle.HIGHEST_PROTOCOL)
    def load_data_(self):
        'load saved data of node to memory'
        if not self.loaded_:
            self.__dict__.update(pickle.load(open(self.pickle_file)))
            self.loaded_ = True
    def unload_data_(self):
        'Remove attributes other than those needed to make'
        if not self.loaded_:
            return
        odict = self.__dict__
        for k in odict.keys():
            if k not in self.make_attributes_:
                del odict[k]
        self.loaded_ = False
    def get_data(self, *args, **kargs):
        self.load_data_()
        return self.decorate_data(*args, **kargs)
    def decorate_data(self, *args, **kargs):
        '''
        represent self for user, should only restruct self
        '''
        raise NotImplementedError('decorate_data is not implemented')
'''
TODO
Node.make works perfectly if in each update, data in dependencies are load from disk.
However if a parent is PickleNode, the data may alreadly in memory.
But when data is large, we want to utilize memory intelligently:
load data once, and unload it when no longer needed.
Thus arises a need for a planner for make
'''
