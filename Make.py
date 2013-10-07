import os
import cPickle as pickle
import time
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
        for parent in self.dependencies:
            parent.make()
        dependency_files = [f for d in self.dependencies for f in d.target_files]
        # NOTE
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
    def __str__(self):
        dependency_files = [d.target_files for d in self.dependencies]
        return "targets: %s, dependencies: %s" % (self.target_files, str(dependency_files))
class RootNode(Node):
    def __init__(self, target_files):
        self.dependencies = []
        self.target_files = target_files
    def update(self):
        for f in self.target_files:
            if not os.path.exists(f):
                raise RuntimeError('root node target %s not exist' % f)
class PhonyNode(Node):
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
    def __init__(self, name, f, dependencies):
        self.name = name
        self.file = f
        self.target_files = [f]
        self.dependencies = dependencies
    def compute(self):
        '''
        do computation and store result in self
        '''
        raise NotImplementedError('compute function is not implemented')
    def update(self):
        print("Update %s" % self.name)
        start_time = time.time()
        self.compute()
        end_time = time.time()
        self.time_used = end_time - start_time
        print("Elapsed time was %g seconds" % self.time_used)
        pickle.dump(self, open(self.target_files[0], 'wb'), pickle.HIGHEST_PROTOCOL)
