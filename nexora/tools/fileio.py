from .basetool import BaseTool
import os
import mimetypes

class FileIO(BaseTool):
    name = "fileio"

    def list(self, path):
        """
        openai.function: This lists the contents of a directory for a given path. You MUST specify a path, but you can use relative path names; for example, `.` will list the current directory, as seen in the primary system prompt. 

        path

        :param str path: The path to list.
        """
        dir = []

        for name in os.listdir(path):
            full_path = os.path.join(path, name)

            if os.path.isdir(full_path):
                type = "dir"
            else:
                type = "file"

            dir.append({
                "type": type,
                "name": name
            })

        return dir
    
    def read(self, path, line_numbers=False):
        """
        openai.function: Read the contents of a file. If you are unsure of what names exist, use fileio_list. You MUST specify the `path` of the file to read. Output is NOT shown to the user; only you see this! 

        path

        :param str path: The path of the file to read.
        :param bool line_numbers: Whether or not to read with line numbers activated. This is useful IF you are going to use the fileio_edit function to modify/patch a part of the file. If you use fileio_write, NEVER bake the line numbers into the file itself, unless instructed explicitly to do so.
        """
        
        with open(path, "r") as f:
            if line_numbers:
                return '\n'.join([f'{i+1} | {line}' for i, line in enumerate(f.read().split('\n'))])
            else:
                return f.read()
        
    def create(self, path):
        """
        openai.function: Create a new, empty file at the specified path. If your intention is to write data to the file, you SHOULD NOT USE THIS and instead, just use fileio_write. 

        path

        :param str path: The path of the file to create.
        """

        self.safe(f"Create blank file {path}")

        with open(path, "w") as f:
            pass

        return "Successfully created {path}"

    def write(self, path, content):
        """
        openai.function: Writes content to a file at the specified path. YOU MUST INCLUDE THE `path` and `content` in order for this to work. The `content` MUST be the entire file, as if there's an existing file at this path, the whole file will be overwritten. IMPORTANT: You must output the whole file as it should be written, without truncating it.

        path,content

        :param str path: The path of the file to write to.
        :param str content: The content to write to the file.
        """

        self.safe(
            reason=f"Write {len(content)} bytes to file {path}",
            preview=content
        )

        with open(path, "w") as f:
            f.write(content)

        return "Successfully wrote {path}"

    def append(self, path, content):
        """
        openai.function: Appends content to a file at the specified path. You MUST specify the path, and the content you'd like to append, in order for this function to work.

        path,content

        :param str path: The path of the file to append to.
        :param str content: The content to append to the file.
        """

        self.safe(
            reason=f"Append {len(content)} bytes to file {path}", 
            preview=content
        )

        with open(path, "a") as f:
            f.write(content)

        return "Successfully appended to {path}"

    def delete(self, path):
        """openai.function: Delete a file at the specified path.

        path

        :param str path: The path of the file to delete.
        """
        self.safe(f"Delete file {path}")

        try:
            if os.path.isfile(path) or os.path.islink(path):
                os.remove(path)
                print(f'File {path} has been deleted')
            elif os.path.isdir(path):
                os.rmdir(path)
                print(f'Directory {path} has been deleted')
            else:
                print(f'Error: {path} is not a file or directory')
        except Exception as e:
            print(f'Error deleting {path}: {e}')

    def mkdir(self, path):
        """
        openai.function: Create a new directory at the specified path.

        path

        :param str path: The path of the directory to create.
        """

        self.safe(f"Create directory {path}")

        os.makedirs(path, exist_ok=True)

        return f"Successfully created directory {path}"

    def rmdir(self, path):
        """
        openai.function: Delete a directory at the specified path.

        path

        :param str path: The path of the directory to delete.
        """

        self.safe(f"Delete directory {path}")

        os.rmdir(path)

        return f"Successfully deleted directory {path}"

    def move(self, source, destination):
        """
        openai.function: Move or rename a file/directory from the source path to the destination path.

        source,destination

        :param str source: The source path of the file/directory to move.
        :param str destination: The destination path where the file/directory will be moved to.
        """

        self.safe(f"Move {source} to {destination}")

        os.rename(source, destination)

        return f"Successfully moved {source} to {destination}"

    def cd(self, path):
        """openai.function: Change the current working directory of the program.

        path

        :param str path: The path to change the working directory to.
        """
        
        os.chdir(path)
        return f"Changed working directory to {path}"

    def walk(self, directory):
        """
        openai.function: Walk the directory tree and return a formatted filesystem tree. You must specify the directory to begin walking.

        directory

        :param str directory: The directory to walk.
        """

        def tree_structure(start_path, level=0):
            tree_str = ''
            for root, dirs, files in os.walk(start_path):
                indent = '    ' * level
                tree_str += f'{indent}+ {root}]\n'
                for dir in sorted(dirs):
                    tree_str += f'{indent}    - {dir}\n'
                for file in sorted(files):
                    tree_str += f'{indent}    * {file}\n'
                level += 1
            return tree_str

        return tree_structure(directory)

    def readwalk(self, directory):
        """
        WALKS AND READS through the directories and files starting from 'directory'. Outputs the contents of all text files, excluding binary and hidden files. You must specify a directory to walk.

        directory

        :param str directory: The root directory to start walking through.
        """
        def is_text_file(filepath):
            type, _ = mimetypes.guess_type(filepath)
            return type is None or type.startswith('text')
        
        def is_hidden(filepath):
            return os.path.basename(filepath).startswith('.')

        output = []

        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not is_hidden(os.path.join(root, d))]  # Exclude hidden directories
            for file in files:
                if not is_hidden(file):
                    file_path = os.path.join(root, file)
                    if is_text_file(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            output.append(f'<content path="{file_path}">{content}</content>')
                        except UnicodeDecodeError:
                            # This handles the rare case where mimetypes considers a binary file as text.
                            pass

        return '\n'.join(output)
