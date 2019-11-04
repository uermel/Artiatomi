function [status, result] = run(command, nodes, config, varargin)
% artia.mpi.run executes Artiatomi tools using mpiexec locally orr
% remotely.
%
% Parameters:
%   command (str):
%       The command to execute.
%   nodes (int):
%       The number of nodes to run the command on.
%   config (str):
%       Path to the config file for this command.
%
% Name Value Pairs:
%   suppressOutput (bool):
%       If true, output of the command being run is not printed in the
%       MATLAB command window.
%   runRemote (bool):
%       If true, command is run using ssh on the host provided in Name
%       Value Pair remoteHost. Requires passwordless ssh setup.
%   remoteHost (str):
%       The remote host to run the command on.
%
% Returns:
%   status (int): 
%       The status code returned from the execution.
%   result (str):
%       The standard output of the command.
%
% Author:
%   UE, 2019

    % Default params
    defs = struct();
    defs.suppressOutput.val = true;
    defs.runRemote.val = false;
    defs.remoteHost.val = '';
    artia.sys.getOpts(varargin, defs);
    
    if runRemote
        com = sprintf('ssh -t %s mpiexec -n %d %s -u %s', remoteHost, nodes, command, config);
    else
        com = sprintf('mpiexec -n %d %s -u %s', nodes, command, config);
    end
    
    if suppressOutput
        [status, result] = system(com);
    else
        [status, result] = system(com)
    end
end

