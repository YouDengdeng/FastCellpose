clear
clc
file_root = 'H:\Code\Python_code\zzz_cellpose_prj\data\train\';
% file_root = 'H:\Code\Python_code\zzz_cellpose_prj\data\test\';
file_Origin = [file_root , 'origin\'];% whole_img 
file_wholemask = [file_root , 'whole_mask\'];% whole_mask
file_inputpic = [file_root , 'input\'];% patch_input
file_maskpic =[file_root , 'mask\'];% patch_mask
img_path_list = dir(strcat(file_Origin,'*.tif'));
img_num = length(img_path_list);
patch_size = [256, 256];
stride = [192, 192];
bina_threshold = 1e-9;

if ~exist(file_inputpic,'dir')
	mkdir(file_inputpic);
end

if ~exist(file_maskpic,'dir')
	mkdir(file_maskpic);
end


if img_num > 0 %有满足条件的图像  
    list_image_real = [];
    list_mask_real = [];
    list_image_unreal = [];
    list_mask_unreal = [];
    index = 0;
    for j = 1:img_num %逐一读取图像  
        disp(j);
        image_name = img_path_list(j).name;% 图像名
        img_origin = imread(strcat(file_Origin,image_name));
        img_mask = imread(strcat(file_wholemask,image_name));
        for col = 0:stride(2):size(img_origin,2)-patch_size(2)
            for row = 0:stride(1):size(img_origin,1)-patch_size(1)
                temp_image = img_origin(row+1:row+patch_size(1),col+1:col+patch_size(2));
                temp_mask = img_mask(row+1:row+patch_size(1),col+1:col+patch_size(2)); 
                if mean(mean(temp_image))>=8  
                    if mean(mean(temp_mask))>=0.0001  
                        index=index+1;
                        list_image_real=cat(3,list_image_real,temp_image);
                        list_mask_real=cat(3,list_mask_real,temp_mask);
                    else
                        list_image_unreal=cat(3,list_image_unreal,temp_image);
                        list_mask_unreal=cat(3,list_mask_unreal,temp_mask);
                    end
                end
            end
        end
    end
    if size(list_mask_real,3)<4*size(list_mask_unreal,3)
        num = floor(size(list_mask_real,3)/4);
        randi = randperm(size(list_mask_unreal,3),num);
        list_image = cat(3,list_image_real,list_image_unreal(:,:,randi));
        list_mask = cat(3,list_mask_real,list_mask_unreal(:,:,randi));
    else
        list_image = cat(3,list_image_real,list_image_unreal);
        list_mask = cat(3,list_mask_real,list_mask_unreal);
    end
%     imwrite(list_image,strcat(file_maskpic,sprintf('input.tif')));
%     imwrite(list_mask,strcat(file_inputpic,sprintf('mask.tif')));
    for i = 1:size(list_image,3)
        name_image = strcat(file_inputpic,sprintf('%d.tif',i));
        name_mask = strcat(file_maskpic,sprintf('%d.tif',i));
        imwrite(list_image(:,:,i),name_image);
        imwrite(list_mask(:,:,i),name_mask);
    end
end