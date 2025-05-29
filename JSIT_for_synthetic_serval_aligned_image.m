%% Load simulated image and codebook
Ic = struct2cell(load('/projects/molonc/scratch/jtsui/MERFISH_synthetic_dataset/results_perfect_dataset_for_jsit_deepcellspots/merfish_perfect_dataset_for_jsit_deepcellspots/1/JSIT_input/serval_aligned_images_001_00.tiff.mat'));
Cc = struct2cell(load('/projects/molonc/scratch/jtsui/JSIT_experiment/reformatted_for_JSIT/XP6873/codebook.mat'));
I = Ic{:};
C = double(Cc{:});  % Ensure proper format

%% Use full image directly
Ic = I;
[sz1, sz2, ~] = size(Ic);

%% Set decoding parameters (L0-tuned)
sf = 3;
sigma = 1.25;
s1 = 40;
s2 = 40;
kmax = 10;
lambda = 5;
k = 5;
t = 0.01;
proxOp = 'L0';
alpha = 0.02;
beta = 200;

%% Print size sanity check
fprintf('Codebook shape: %s\n', mat2str(size(C)));
fprintf('Image stack shape: %s\n', mat2str(size(Ic)));

%% Prepare matrices for FISTA
A = getPsfMat2(40 * sf, sf, sigma);
K = C * C'; [~, Sk, ~] = svd(K); eK = Sk(1,1);
Ma = A * A'; [~, Sm, ~] = svd(Ma); eM = Sm(1,1);
A = sparse(A);

%% Pre-process image
Ystack = double(makeYstack(Ic, s1, s2));
nYs = size(Ystack, 3);
Xstack = zeros(nYs, (s1 * sf)^2, size(C, 1));

%% Run FISTA decoding (L0 mode)
fprintf('Running FISTA (L0)...\n');
for x = 1:nYs
    Ypatch = Ystack(:,:,x);
    Xstack(x,:,:) = codebookFISTA(A, C, Ypatch, lambda, kmax, proxOp, alpha, beta, eK, eM);
end

%% Post-process decoding results
d1 = sz1 / s1;
d2 = sz2 / s2;
X = processXstack(Xstack, d1, d2, s1 * sf, s1 * sf);
Xf = enforceSparsity2(X, k, t);

%% Debug: Check matrix activity
fprintf('Non-zero entries in Xf: %d / %d\n', nnz(Xf), numel(Xf));

xIm = reshape(sum(Xf, 2), [sz1 * sf, sz2 * sf]);
Icv = reshape(Ic, [sz1 * sz2, size(Ic, 3)]);
iv = sqrt(sum(double(Icv).^2, 2));
iIm = reshape(iv, [sz1, sz2]);
iIm = imresize(iIm, sf);

[dIm, xIm_vals] = X2dIm(Xf, sz1 * sf, sz2 * sf);
q = dIm2q_ex(dIm, iIm, xIm_vals, 2, C);
q(:,1:2) = q(:,1:2) ./ sf;

%% Debug: Check transcript calls
fprintf('Decoded transcript calls: %d\n', size(q,1));

%% TEMP: Skip adaptive filtering
qqt = q;
qqct = q(q(:,4) <= 115, :);

%% Save CSVs
output_folder = '/projects/molonc/scratch/jtsui/JSIT_experiment/output/synthetic';
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

fid = fopen(fullfile(output_folder, 'barcodes_with_blanks.csv'), 'w');
fprintf(fid, 'x,y,barcode_id,spot_area,image_intensity,decoded_signal_strength\n');
fclose(fid);
dlmwrite(fullfile(output_folder, 'barcodes_with_blanks.csv'), qqt, '-append');

fid = fopen(fullfile(output_folder, 'barcodes_wo_blanks.csv'), 'w');
fprintf(fid, 'x,y,barcode_id,spot_area,contrast,confidence_score\n');
fclose(fid);
dlmwrite(fullfile(output_folder, 'barcodes_wo_blanks.csv'), qqct, '-append');

%% Save decoded intensity map as 16-bit TIFF
xIm_scaled = xIm - min(xIm(:));
xIm_scaled = xIm_scaled / max(xIm_scaled(:));         % Normalize to [0,1]
xIm_uint16 = uint16(xIm_scaled * 65535);              % Convert to 16-bit

tiff_path = fullfile(output_folder, 'decoded_intensity_map_uint16.tiff');
imwrite(xIm_uint16, tiff_path);

fprintf('Saved 16-bit decoded transcript map to: %s\n', tiff_path);
disp('Decoding complete and outputs saved.');

