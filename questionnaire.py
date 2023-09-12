from tkinter import *
from constants import QUESTIONS_LIST,QUESTIONNAIRE_HEADING
import utils


class SubWindow(Toplevel):
    def __init__(self, master):
        super().__init__(master)

        self.title("Question List")
        ans_array = [None] * len(QUESTIONS_LIST)
        lbl_answers = {}

        def btn_pressed(index, ans='No'):
            print(index)
            ans_array[index] = ans
            lbl_answers[index].configure(text=ans)
            print(ans_array)

        def add_new(data, row, index):
            lbl_question = Label(self, text=data)
            lbl_answers[index] = Label(self)

            btn_yes = Button(self, text='Yes', command=lambda: btn_pressed(index, 'Yes'))
            btn_no = Button(self, text='No', command=lambda: btn_pressed(index, 'No'))

            lbl_question.grid(row=row, sticky=W)
            lbl_answers[index].grid(column=0, row=row + 1, sticky=E)
            btn_yes.grid(column=1, row=row + 1, sticky=E)
            btn_no.grid(column=2, row=row + 1, sticky=E)

        def generate_report():
            if any(x is None for x in ans_array):
                print('error')
            else:
                utils.save_sensitivity_results_to_csv(ans_array, '../results/questionnaire_result.csv')
                self.destroy()

        def show_questions():
            row = 2
            index = 0
            for x in QUESTIONS_LIST:
                row = row + 2
                add_new(x, row, index)
                index += 1
            btn_submit = Button(self, text='Submit', command=generate_report, width=50)
            btn_submit.grid(column=0, row=row + 4, sticky=W, columnspan=3)

        Label(self, text=QUESTIONNAIRE_HEADING, font=("Calibri", 14)).grid(column=0, columnspan=3, row=0, sticky=W)
        show_questions()
        self.mainloop()




