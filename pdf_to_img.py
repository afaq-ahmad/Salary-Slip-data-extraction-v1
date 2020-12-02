from pdf2image import convert_from_path
import os


def pdf_to_image(input_pdf_dir, image_output_dir):
    if not os.path.isdir(image_output_dir):
        os.mkdir(image_output_dir)
        print("directory created\n", image_output_dir)
    pdf_files = os.listdir(input_pdf_dir)

    for pdf_file in pdf_files:
        if ".pdf" in pdf_file:
            input_pdf_file = input_pdf_dir + pdf_file
            print(pdf_file)
            pages = convert_from_path(input_pdf_file, 300,poppler_path="C:\\poppler-0.68.0\\bin") # 100 gives almost accurate coords
            for i, page in enumerate(pages):
                output_file = (''.join(input_pdf_file.split('/')[-1])).replace(" ", "")
                output_file = '.'.join(output_file.split('.')[:-1]) + "-page%s.pdf.png" % i
                image_output_path = image_output_dir + output_file
                page.save(image_output_path, 'PNG')
    return


def main():
    input_pdf_dir = "pdf_dir/"
    image_output_dir = "output_img_dir/"
    pdf_to_image(input_pdf_dir, image_output_dir)
    return


if __name__ == "__main__":
    main()
