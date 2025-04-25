function [] = JSIT(fov, codebook, predicted_folder)

    %% Extract dataset name from FOV path
    [path_to_fov, ~, ~] = fileparts(fov);
    [~, dataset_name] = fileparts(path_to_fov);
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
    C = double(Cc{:});

    cropSize = 400;  % changed from 402 to 400 for divisibility
    numBlocksX = ceil(W / cropSize);
    numBlocksY = ceil(H / cropSize);

    % Accumulators for final outputs
    qqt_dict = containers.Map('KeyType', 'char', 'ValueType', 'any');
    qqct_dict = containers.Map('KeyType', 'char', 'ValueType', 'any');
    %% xIm_full = zeros(H * sf, W * sf);
    %% dIm_full = zeros(H * sf, W * sf);
    %% iIm_full = zeros(H * sf, W * sf);

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

            % Set parameters
            s1 = 40; s2 = 40; kmax = 10; lambda = 75; k = 1; t = 0;

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

            % FISTA decoding
            for x = 1:nYs
                Xstack(x, :, :) = codebookFISTA(A, C, Ystack(:, :, x), lambda, kmax, 'SGL', 0.5, [], eK, eM);
            end

            % Postprocess
            d1 = sz1 / s1; d2 = sz2 / s2;
            X = processXstack(Xstack, d1, d2, s1 * sf, s1 * sf);
            Xf = enforceSparsity2(X, k, t);
            xIm = reshape(sum(Xf, 2), [sz1 * sf, sz2 * sf]);
            Icv = reshape(Ic, [sz1 * sz2, size(Ic, 3)]);
            iv = sqrt(sum(double(Icv).^2, 2));
            iIm = reshape(iv, [sz1, sz2]);
            iIm = imresize(iIm, sf);
            dIm = X2dIm(Xf, sz1 * sf, sz2 * sf);
            q = dIm2q_ex(dIm, iIm, xIm, 2, C);
            q(:,1:2) = q(:,1:2) ./ sf;

            % Adaptive filtering
            nA = 10; nI = 10; nX = 10; nC = 115; nB = 25; tgt = 0.05;
            [qqt, qqct] = getThresholdHist(q, nA, nI, nX, nC, nB, tgt);

            key = sprintf('%d_%d', block_x, block_y);
            qqt_dict(key) = qqt;
            qqct_dict(key) = qqct;

            % Stitch tile into full image
            %% y_idx = y_start:y_end;
            %% x_idx = x_start:x_end;
            %% xIm_full(y_idx, x_idx) = xIm;
            %% iIm_full(y_idx, x_idx) = iIm;
            %% dIm_full(y_idx, x_idx) = dIm;
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
    fprintf(fid, 'x,y,barcode_id,spot_area,col5,probability\n');
    fclose(fid);
    dlmwrite(fullfile(predicted_folder, 'barcodes_with_blanks.csv'), qqt_all, '-append');
    fid = fopen(fullfile(predicted_folder, 'barcodes_wo_blanks.csv'), 'w');
    fprintf(fid, 'x,y,barcode_id,spot_area,col5,probability\n');
    fclose(fid);
    dlmwrite(fullfile(predicted_folder, 'barcodes_wo_blanks.csv'), qqct_all, '-append');

    disp('--- All blocks processed and saved ---');
    exit;
end
