function [] = JSIT_serval_aligned(fov, codebook, predicted_folder)

    %% Extract dataset name from FOV path
    [path_to_fov, ~, ~] = fileparts(fov);
    [~, dataset_name] = fileparts(path_to_fov);
    disp(['Detected dataset: ', dataset_name]);

    %% Assign sf and sigma based on dataset
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

    %% Load data
    Ic = struct2cell(load(fov));
    Cc = struct2cell(load(codebook));
    I = double(Ic{:});
    C = double(Cc{:});

    % Pad image to 2000 x 2000 if needed
    padded_H = 2000;
    padded_W = 2000;
    [H, W, D] = size(I);
    if H < padded_H || W < padded_W
        padded_I = zeros(padded_H, padded_W, D);
        padded_I(1:H, 1:W, :) = I;
        I = padded_I;
        H = padded_H;
        W = padded_W;
        disp(['Image padded to ', num2str(H), ' x ', num2str(W)]);
    end

    cropSize = 400;  % changed from 402 to 400 for divisibility
    numBlocksX = ceil(W / cropSize);
    numBlocksY = ceil(H / cropSize);

    % Accumulators for final outputs
    qqt_dict = containers.Map('KeyType', 'char', 'ValueType', 'any');
    qqct_dict = containers.Map('KeyType', 'char', 'ValueType', 'any');
    %% xIm_full = zeros(H * sf, W * sf);
    %% dIm_full = zeros(H * sf, W * sf);
    %% iIm_full = zeros(H * sf, W * sf);
    xIm_full = zeros(H * sf, W * sf);

    for block_x = 0:numBlocksX-1
        for block_y = 0:numBlocksY-1
            x_start = block_x * cropSize + 1;
            y_start = block_y * cropSize + 1;
            x_end = min((block_x + 1) * cropSize, W);
            y_end = min((block_y + 1) * cropSize, H);

            % Skip partial tiles at the edges
            if (x_end - x_start + 1 < cropSize) || (y_end - y_start + 1 < cropSize) || (x_end > W) || (y_end > H)
                continue;
            end

            disp(['Decoding block covering image region (X: ', num2str(x_start), '-', num2str(x_end), ', Y: ', num2str(y_start), '-', num2str(y_end), ')']);

            Ic = I(y_start:y_end, x_start:x_end, :);
            sz1 = size(Ic, 1);
            sz2 = size(Ic, 2);

            % Prepare matrices
            A = getPsfMat2(40 * sf, sf, sigma);
            K = C * C'; [~, Sk, ~] = svd(K); eK = Sk(1, 1);
            Ma = A * A'; [~, Sm, ~] = svd(Ma); eM = Sm(1, 1);
            A = sparse(A);

            % Preprocess
            Ih = imgaussfilt(Ic, 0.5) - imgaussfilt(Ic, 3);
            Ystack = double(makeYstack(Ih, s1, s2));
            nYs = size(Ystack, 3);
            Xstack = zeros(nYs, (s1 * sf)^2, size(C, 1));

            %% Run FISTA decoding (L0 mode)
            fprintf('Running FISTA (L0)...\n');
            for x = 1:nYs
                Ypatch = Ystack(:,:,x);
                Xstack(x,:,:) = codebookFISTA(A, C, Ypatch, lambda, kmax, proxOp, alpha, beta, eK, eM);
            end

            % Postprocess
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

            key = sprintf('%d_%d', block_x, block_y);
            qqt_dict(key) = qqt;
            qqct_dict(key) = qqct;

            % Stitch tile into full image
            y_idx = (y_start-1)*sf + 1 : (y_start-1 + sz1)*sf;
            x_idx = (x_start-1)*sf + 1 : (x_start-1 + sz2)*sf;
            xIm_full(y_idx, x_idx) = xIm;

        end
    end

    % Postprocess qqt and qqct for global coordinates
    qqt_all = [];
    qqct_all = [];
    keys_list = keys(qqt_dict);
    for i = 1:length(keys_list)
        key = keys_list{i};
        parts = sscanf(key, '%d_%d');
        block_x = parts(1);
        block_y = parts(2);

        q1 = qqt_dict(key);
        q2 = qqct_dict(key);
        q1(:,1) = q1(:,1) + block_x * cropSize;
        q1(:,2) = q1(:,2) + block_y * cropSize;
        q2(:,1) = q2(:,1) + block_x * cropSize;
        q2(:,2) = q2(:,2) + block_y * cropSize;

        qqt_all = [qqt_all; q1];
        qqct_all = [qqct_all; q2];
    end

    % Save final results
    if ~exist(predicted_folder, 'dir')
        mkdir(predicted_folder);
    end

    %% imwrite(dIm_full, fullfile(predicted_folder, 'unfiltered_predictions.tiff'));
    %% imwrite(iIm_full, fullfile(predicted_folder, 'spot_predictions.tiff'));
    fid = fopen(fullfile(predicted_folder, 'barcodes_with_blanks.csv'), 'w');
    fprintf(fid, 'x,y,barcode_id,spot_area,image_intensity,decoded_signal_strength\n');
    fclose(fid);
    dlmwrite(fullfile(predicted_folder, 'barcodes_with_blanks.csv'), qqt_all, '-append');
    fid = fopen(fullfile(predicted_folder, 'barcodes_wo_blanks.csv'), 'w');
    fprintf(fid, 'x,y,barcode_id,spot_area,contrast,confidence_score\n');
    fclose(fid);
    dlmwrite(fullfile(predicted_folder, 'barcodes_wo_blanks.csv'), qqct_all, '-append');

    % Normalize xIm_full to [0, 1] and convert to 16-bit
    xIm_scaled = xIm_full - min(xIm_full(:));
    xIm_scaled = xIm_scaled / max(xIm_scaled(:));
    xIm_uint16 = uint16(xIm_scaled * 65535);
    
    % Downsample to 2000 x 2000
    xIm_resized = imresize(xIm_uint16, [2000 2000]);
    
    % Save as TIFF
    tiff_path = fullfile(predicted_folder, 'decoded_intensity_map_uint16.tiff');
    imwrite(xIm_resized, tiff_path);
    
    fprintf('Saved 16-bit decoded transcript map to: %s\n', tiff_path);
    disp('Decoding complete and outputs saved.');


    disp('--- All blocks processed and saved ---');
    exit;
end
