%% Load simulated image and codebook
Ic = struct2cell(load('/projects/molonc/scratch/jtsui/MERFISH_synthetic_dataset/results_perfect_dataset_for_jsit_deepcellspots/merfish_perfect_dataset_for_jsit_deepcellspots/1/JSIT_input/serval_aligned_preprocessed_images_001_00.tiff.mat'));
Cc = struct2cell(load('/projects/molonc/scratch/jtsui/JSIT_experiment/reformatted_for_JSIT/XP6873/codebook.mat'));
I = Ic{:};
C = Cc{:};
C = double(Cc{:});  % <-- Fix applied here

%% Use full image directly (assumes it's 400 x 400)
Ic = I;
[sz1, sz2, ~] = size(Ic);

%% Set decoding parameters
sf = 3;
sigma = 1.25;
s1 = 40;
s2 = 40;
kmax = 10;
lambda = 25;
k = 2;
t = 0;

%% Prepare matrices for FISTA
A = getPsfMat2(40*sf, sf, sigma);
K = C * C'; [~, Sk, ~] = svd(K); eK = Sk(1,1);
Ma = A * A'; [~, Sm, ~] = svd(Ma); eM = Sm(1,1);
A = sparse(A);

%% Pre-process image
Ih = Ic;
Ystack = double(makeYstack(Ih, s1, s2));
nYs = size(Ystack, 3);
Xstack = zeros(nYs, (s1 * sf)^2, size(C, 1));

%% Run FISTA decoding
for x = 1:nYs
    Xstack(x,:,:) = codebookFISTA(A, C, Ystack(:,:,x), lambda, kmax, 'SGL', 0.5, [], eK, eM);
end

%% Post-process decoding results
d1 = sz1 / s1;
d2 = sz2 / s2;
X = processXstack(Xstack, d1, d2, s1 * sf, s1 * sf);
Xf = enforceSparsity2(X, k, t);
xIm = reshape(sum(Xf, 2), [sz1 * sf, sz2 * sf]);
Icv = reshape(Ic, [sz1 * sz2, size(Ic, 3)]);
iv = sqrt(sum(double(Icv).^2, 2));
iIm = reshape(iv, [sz1, sz2]);
iIm = imresize(iIm, sf);
dIm = X2dIm(Xf, sz1 * sf, sz2 * sf);
q = dIm2q_ex(dIm, iIm, xIm, 2, C);
q(:,1:2) = q(:,1:2) ./ sf;  % Convert back to original coordinates

%% Adaptive thresholding
nA = 10; nI = 10; nX = 10; nC = 115; nB = 25; tgt = 0.05;
[qqt, qqct] = getThresholdHist(q, nA, nI, nX, nC, nB, tgt);

%% Save CSVs to output folder
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

disp('✅ Decoding complete and spot tables saved.');

