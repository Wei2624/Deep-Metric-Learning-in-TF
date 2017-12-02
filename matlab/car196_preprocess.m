cars_train_annos = load('devkit/cars_train_annos.mat');
cars_train_annos = cars_train_annos.annotations;

train_flag = true;

images = struct();
pos = 0;
crop_padding = 15;
force_square_size = 256;

base_dir = '/home/xuzhang/project/Medifor/data/car196/';

for i_img = 1:length(cars_train_annos)
    if train_flag
        if cars_train_annos(i_img).class > 98
            continue;
        end
    else
        if cars_train_annos(i_img).class <= 98
            continue;
        end
    end
    filename = fullfile(strcat(base_dir,'cars_train/'), cars_train_annos(i_img).fname);
    orig_img = imread(filename);
    if (ndims(orig_img) >=3)
        view = rgb2gray(orig_img);
    else
        % if read as gray scale, convert orig image to rgb
        view = orig_img;
        orig_img = cat(3, view, view, view);
    end
    [row, col] = find(view < 250);
    xmin = max(0, min(col) - crop_padding);
    xmax = min(size(view, 2), max(col) + crop_padding);
    ymin = max(0, min(row) - crop_padding);
    ymax = min(size(view, 1), max(row) + crop_padding);
    
    width = xmax - xmin + 1;
    height = ymax - ymin + 1;
    img_cropped = imcrop(orig_img, [xmin, ymin, width, height]);

    % hos: recompute width, height (sometimes gets cropped out of bound)
    [height, width, dd] = size(img_cropped);
    
    if force_square_size > 0
        if height > width
            img_cropped = imresize(img_cropped, [force_square_size NaN]);
            pre_padding = floor((force_square_size - size(img_cropped, 2)) / 2);
            img_cropped = padarray(img_cropped, [0 pre_padding 0], 255, 'pre');
            post_padding = force_square_size - size(img_cropped, 2);
            img_cropped = padarray(img_cropped, [0 post_padding 0], 255, 'post');
        elseif width > height
            img_cropped = imresize(img_cropped, [NaN force_square_size]);
            pre_padding = floor((force_square_size - size(img_cropped, 1)) / 2);
            img_cropped = padarray(img_cropped, [pre_padding 0 0], 255, 'pre');
            post_padding = force_square_size - size(img_cropped, 1);
            img_cropped = padarray(img_cropped, [post_padding 0 0], 255, 'post');
        else
            img_cropped = imresize(...
                img_cropped, [force_square_size force_square_size]);
        end
        assert(size(img_cropped, 1) == ...
            force_square_size && size(img_cropped, 2) == force_square_size);
    end
    
    pos = pos + 1;
    images(pos).img = img_cropped;
    images(pos).crop_bbox = [ymin, xmin, ymax, xmax];
    images(pos).filename = filename;
    images(pos).class_id = cars_train_annos(i_img).class;
    
    fprintf('Loaded %d/%d images(%.1f%%)\n', pos, length(cars_train_annos), ...
        100*pos / length(cars_train_annos));
end

cars_test_annos = load('devkit/cars_test_annos_withlabels.mat');
cars_test_annos = cars_test_annos.annotations;

for i_img = 1:length(cars_test_annos)
    if train_flag
        if cars_test_annos(i_img).class > 98
            continue;
        end
    else
        if cars_test_annos(i_img).class <= 98
            continue;
        end
    end
    filename = fullfile(strcat(base_dir,'cars_test/'), cars_test_annos(i_img).fname);
    orig_img = imread(filename);
    if (ndims(orig_img) >=3)
        view = rgb2gray(orig_img);
    else
        % if read as gray scale, convert orig image to rgb
        view = orig_img;
        orig_img = cat(3, view, view, view);
    end
    [row, col] = find(view < 250);
    xmin = max(0, min(col) - crop_padding);
    xmax = min(size(view, 2), max(col) + crop_padding);
    ymin = max(0, min(row) - crop_padding);
    ymax = min(size(view, 1), max(row) + crop_padding);
    
    width = xmax - xmin + 1;
    height = ymax - ymin + 1;
    img_cropped = imcrop(orig_img, [xmin, ymin, width, height]);

    % hos: recompute width, height (sometimes gets cropped out of bound)
    [height, width, dd] = size(img_cropped);
    
    if force_square_size > 0
        if height > width
            img_cropped = imresize(img_cropped, [force_square_size NaN]);
            pre_padding = floor((force_square_size - size(img_cropped, 2)) / 2);
            img_cropped = padarray(img_cropped, [0 pre_padding 0], 255, 'pre');
            post_padding = force_square_size - size(img_cropped, 2);
            img_cropped = padarray(img_cropped, [0 post_padding 0], 255, 'post');
        elseif width > height
            img_cropped = imresize(img_cropped, [NaN force_square_size]);
            pre_padding = floor((force_square_size - size(img_cropped, 1)) / 2);
            img_cropped = padarray(img_cropped, [pre_padding 0 0], 255, 'pre');
            post_padding = force_square_size - size(img_cropped, 1);
            img_cropped = padarray(img_cropped, [post_padding 0 0], 255, 'post');
        else
            img_cropped = imresize(...
                img_cropped, [force_square_size force_square_size]);
        end
        assert(size(img_cropped, 1) == ...
            force_square_size && size(img_cropped, 2) == force_square_size);
    end
    
    pos = pos + 1;
    images(pos).img = img_cropped;
    images(pos).crop_bbox = [ymin, xmin, ymax, xmax];
    images(pos).filename = filename;
    images(pos).class_id = cars_test_annos(i_img).class;
    
    fprintf('Loaded %d/%d images(%.1f%%)\n', pos, length(cars_test_annos), ...
        100*pos / length(cars_test_annos));
end

validation_images = images;

if train_flag
    training_images = validation_images;
    savefast(fullfile(base_dir,'training_cars196_256cropped.mat'), 'training_images');
else
    savefast(fullfile(base_dir,'validation_cars196_256cropped.mat'), 'validation_images');
end
