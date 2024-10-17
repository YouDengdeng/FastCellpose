clear
clc
% file_root = 'H:\Code\Python_code\zzz_cellpose_prj\data\train\';
file_root = 'H:\Code\Python_code\zzz_cellpose_prj\data\test\';
file_Origin = [file_root , 'origin\'];% whole_img 
file_wholemask = [file_root , 'whole_mask\'];% whole_mask
file_inputpic = [file_root , 'input\'];% patch_input
file_maskpic =[file_root , 'mask\'];% patch_mask
img_path_list = dir(strcat(file_Origin,'*.tif'));
img_num = length(img_path_list);
patch_size = [256, 256];
stride = [192, 192];
bina_threshold = 1e-9;

input_sig_thresh = 4;
mask_sig_thresh = 1;

if ~exist(file_inputpic,'dir')
	mkdir(file_inputpic);
end

if ~exist(file_maskpic,'dir')
	mkdir(file_maskpic);
end


if img_num > 0 
    list_image_real = [];
    list_mask_real = [];
    list_image_unreal = [];
    list_mask_unreal = [];
    index = 0;
    for j = 1:img_num 
        disp(j);
        image_name = img_path_list(j).name;
        img_origin = imread(strcat(file_Origin,image_name));
        img_mask = 255*imread(strcat(file_wholemask,image_name));
        for col = 0:stride(2):size(img_origin,2)-patch_size(2)
            for row = 0:stride(1):size(img_origin,1)-patch_size(1)
                temp_image = img_origin(row+1:row+patch_size(1),col+1:col+patch_size(2));
                temp_mask = img_mask(row+1:row+patch_size(1),col+1:col+patch_size(2)); 
                if mean(mean(temp_image))>=input_sig_thresh  %input signal>thresh
%                 if (mean(mean(temp_mask))>=0.01)  &&  (mean(mean(temp_mask))<=1)
                    if mean(mean(temp_mask))>=mask_sig_thresh  %mask signal>thresh
                        name_image = strcat(file_inputpic,sprintf('%d.tif',index));
                        name_mask = strcat(file_maskpic,sprintf('%d.tif',index));
                        imwrite(temp_image,name_image);
                        imwrite(temp_mask,name_mask);
                        index = index +1;
                    end
                end
            end
        end
    end
end