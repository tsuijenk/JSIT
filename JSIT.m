function [] = JSIT(fov,codebook,predicted_folder)

    %% Example Script for running JSIT
    
    %% Extract dataset name from FOV path
    % Example input: /projects/molonc/scratch/jtsui/JSIT_experiment/reformatted_for_JSIT/XP6873/aligned_images_000_01.tiff.mat

    [path_to_fov, ~, ~] = fileparts(fov);       % gets: .../XP6873
    [~, dataset_name] = fileparts(path_to_fov); % gets: XP6873
    
    disp(['Detected dataset: ', dataset_name]);

    %% Assign sf and sigma based on dataset
    if strcmp(dataset_name, 'XP6873')
        sf = 3;
        sigma = 2.3;
    elseif strcmp(dataset_name, 'XP8054')
        sf = 2;
        sigma = 0.3;
    else
        error(['Unknown dataset: ', dataset_name]);
    end
    
    disp(['Using sf = ', num2str(sf), ', sigma = ', num2str(sigma)]);


    %% Load data
    Ic = struct2cell(load(fov));
    Cc = struct2cell(load(codebook));

    disp('--- Loaded input files ---');

    I = double(Ic{:});
    C = double(Cc{:});
    disp(['Input image size: ', mat2str(size(I))]);

    %% Crop image to reduce RAM requirement
    cropSize = 402;
    block_x = 3;
    block_y = 3;
    Ic = I((block_x*cropSize)+1:(block_x+1)*cropSize,(block_y*cropSize)+1:(block_y+1)*cropSize,:);
    sz1 = size(Ic,1);
    sz2 = size(Ic,2);
    disp(['Cropped Ic size: ', mat2str(size(Ic))]);

    %% Set parameters
    %% sf = 3; %Resolution scale factor 
    %% sigma = 1.25; %Width of microscope PSF
    s1 = 40; %Patch size
    s2 = 40;
    kmax = 10; %FISTA iterations
    lambda = 75; %Regularization parameter
    k = 1; %Row-sparsity of estimated X
    t = 0; %Hard-threshold on estimated X

    %% Prepare matrices for FISTA
    disp('--- Preparing PSF and eigenvalue matrices ---');
    A = getPsfMat2(40*sf,sf,sigma);
    K = C*C';
    [~,Sk,~] = svd(K);
    eK = Sk(1,1);
    Ma = A*A';
    [~,Sm,~] = svd(Ma);
    eM = Sm(1,1);
    A = sparse(A);

    %% Pre-process data
    disp('--- Preprocessing and generating Ystack ---');
    Ih = imgaussfilt(Ic,0.5)-imgaussfilt(Ic,3);
    Ystack = double(makeYstack(Ih,s1,s2));
    nYs = size(Ystack,3);
    disp(['Ystack size: ', mat2str(size(Ystack))]);
    Xstack = zeros(nYs,(s1.*sf)^2,size(C,1));
    disp(['Initialized Xstack size: ', mat2str(size(Xstack))]);

    %% Run FISTA
    disp('--- Running FISTA decoding on patches ---');
    for x = 1:nYs
        if mod(x, 10) == 0
            disp(['Processing patch ', num2str(x), ' / ', num2str(nYs)]);
        end
        Xstack(x,:,:) = codebookFISTA(A,C,Ystack(:,:,x),lambda,kmax,'SGL',0.5,[],eK,eM);
    end

    %% Post-process decoding results
    disp('--- Reshaping decoded patches ---');
    d1 = sz1./s1;
    d2 = sz2./s2;
    disp(['Calling processXstack with patch size: ', num2str(s1*sf), ' x ', num2str(s1*sf)]);
    X = processXstack(Xstack,d1,d2,s1*sf,s1*sf);

    Xf = enforceSparsity2(X,k,t);
    xIm = reshape(sum(Xf,2),[sz1*sf,sz2*sf]);
    Icv = reshape(Ic,[sz1*sz2,16]);
    iv = sqrt(sum(double(Icv).^2, 2));
    iIm = reshape(iv,[sz1,sz2]);
    iIm = imresize(iIm,sf);
    dIm = X2dIm(Xf,sz1*sf,sz2*sf);
    q = dIm2q_ex(dIm,iIm,xIm,2,C);
    q(:,1:2) = q(:,1:2)./sf;

    %% Adaptive filtering
    nA = 10;
    nI = 10;
    nX = 10;
    nC = 115;
    nB = 25;
    tgt = 0.05;
    [qqt,qqct] = getThresholdHist(q,nA,nI,nX,nC,nB,tgt);

    if ~exist(predicted_folder, 'dir')
        mkdir(predicted_folder);
    end

    imwrite(dIm,fullfile(predicted_folder,'unfiltered_predictions.tiff'));
    imwrite(iIm,fullfile(predicted_folder,'spot_predictions.tiff'));
    dlmwrite(fullfile(predicted_folder, 'barcodes_with_blanks.csv'), qqt, 'delimiter', ',');
    dlmwrite(fullfile(predicted_folder, 'barcodes_wo_blanks.csv'), qqct, 'delimiter', ',');

    % Verify and print status
    file_list = {
        'unfiltered_predictions.tiff'
        'spot_predictions.tiff'
        'barcodes_with_blanks.csv'
        'barcodes_wo_blanks.csv'
    };

    for i = 1:length(file_list)
        file_path = fullfile(predicted_folder, file_list{i});
        if exist(file_path, 'file')
            fprintf('Successfully saved: %s\n', file_path);
        else
            fprintf('Failed to save: %s\n', file_path);
        end
    end

    %% Explicitly exit the MATLAB runtime to avoid hanging jobs
    exit;

end