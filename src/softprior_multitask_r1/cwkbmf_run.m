% Driver for the authors' cwKBMF code (Ammad-ud-din et al.; file cwkbmf_1mkl1k_regression_variational_*.m,
% used unmodified from the published archive). Reads kernels and standardised responses for a list of
% fits, trains one model per (fit, R) and writes the predicted response matrices.
%   octave --no-gui --quiet --eval "cwkbmf_run('<code dir>', '<in.mat>', '<out.mat>')"
function cwkbmf_run(code_dir, in_file, out_file)
  addpath(code_dir);
  S = load(in_file);
  out = struct();
  Rs = double(S.Rs(:))';
  for k = 1:double(S.n_fits)
    Ktr = S.(sprintf('Ktr_%d', k));
    Kte = S.(sprintf('Kte_%d', k));
    Y = S.(sprintf('Z_%d', k));
    Kz = eye(size(Y, 2));
    for R = Rs
      p = struct();
      p.alpha_lambda = S.alpha_lambda; p.beta_lambda = S.beta_lambda;
      p.alpha_eta = S.alpha_eta; p.beta_eta = S.beta_eta;
      p.iteration = double(S.iteration); p.progress = 0; p.R = R; p.seed = double(S.seed);
      p.sigmag = S.sigmag; p.sigmah = S.sigmah; p.sigmay = S.sigmay;
      state = cwkbmf_1mkl1k_regression_variational_train(Ktr, Kz, Y, p);
      pred = cwkbmf_1mkl1k_regression_variational_test(Kte, Kz, state);
      out.(sprintf('P_%d_R%d', k, R)) = pred.Y.mean;
      out.(sprintf('e_%d_R%d', k, R)) = state.ex.mean;
    end
  end
  save('-v7', out_file, '-struct', 'out');
end
