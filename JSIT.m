function [] = JSIT(fov,codebook,predicted_folder)

    %% Example Script for running JSIT
    
    %% Load data
    Ic = struct2cell(load(fov));
    Cc = struct2cell(load(codebook));
    I = double(Ic{:});
    C = double(Cc{:});
    
    
    %% Crop image to reduce RAM requirement
    cropSize = 400;
    block_x = 3;
    block_y = 3;
    Ic = I((block_x*cropSize)+1:(block_x+1)*cropSize,(block_y*cropSize)+1:(block_y+1)*cropSize,:);
    sz1 = size(Ic,1);
    sz2 = size(Ic,2);
    
    %% Set parameters
    sf = 3; %Resolution scale factor 
    sigma = 1.25; %Width of microscope PSF
    s1 = 40; %Patch size
    s2 = 40;
    kmax = 10; %FISTA iterations
    lambda = 75; %Regularization parameter
    k = 1; %Row-sparsity of estimated X
    t = 0; %Hard-threshold on estimated X
    
    %% Prepare matrices for FISTA
    A = getPsfMat2(40*sf,sf,sigma);
    %Pre-compute largest Eigenvalues of CC' and AA'
    K = C*C';
    [~,Sk,~] = svd(K);
    eK = Sk(1,1);
    Ma = A*A';
    [~,Sm,~] = svd(Ma);
    eM = Sm(1,1);
    A = sparse(A);
    
    %% Pre-process data
    Ih = imgaussfilt(Ic,0.5)-imgaussfilt(Ic,3);
    Ystack = double(makeYstack(Ih,s1,s2));
    nYs = size(Ystack,3);
    Xstack = zeros(nYs,(s1.*sf)^2,size(C,1));
    
    %% Run FISTA
    for x = 1:nYs
        Xstack(x,:,:) = codebookFISTA(A,C,Ystack(:,:,x),lambda,kmax,'SGL',0.5,[],eK,eM);
    end
    
    %% Post-process decoding results
    d1 = sz1./s1;
    d2 = sz2./s2;
    X = processXstack(Xstack,d1,d2,s1*sf,s1*sf);
    Xf = enforceSparsity2(X,k,t);
    xIm = reshape(sum(Xf,2),[sz1*sf,sz2*sf]);
    Icv = reshape(Ic,[sz1*sz2,16]);
    iv = vecnorm(double(Icv)',2)';
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
    
    mkdir(predicted_folder)
    
    imwrite(dIm,fullfile(predicted_folder,'unfiltered_predictions.tiff'));
    imwrite(iIm,fullfile(predicted_folder,'spot_predictions.tiff'));
    writematrix(qqt,fullfile(predicted_folder,'barcodes_with_blanks.csv'));
    writematrix(qqct,fullfile(predicted_folder,'barcodes_wo_blanks.csv'));

end