clear
clc
% file_root = 'H:\Code\Python_code\zzz_cellpose_prj\data\train\';
file_root = 'H:\Code\Python_code\zzz_cellpose_prj\data\test\';

filepath = [file_root, 'mask\'];
savepath = [file_root, 'annotate\'];

if ~exist(savepath,'dir')
	mkdir(savepath);
end

filedir = dir(filepath);
filedir=filedir(3:end);
p = length(filedir);

for i=1:p
    image = imread([filepath filedir(i).name]);
    image_convert = annotate(image);
    imwrite(image_convert,[savepath filedir(i).name])
end