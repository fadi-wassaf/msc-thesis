function [poles, R, fit, rmserr] = VFWrapper(bigH, s, poles, opts)
    % NOTE: This wrapper file should be included within any directory where the vector fitting routine is needed.
    addpath C:\mtrxfit; % NOTE: Should be location of the mtrxfit Package (see https://www.sintef.no/en/software/vector-fitting/downloads/matrix-fitting-toolbox/)
    [SER, rmserr, fit, ~] = VFdriver(bigH, s, poles, opts);
    poles = SER.poles;
    R = SER.R;
end